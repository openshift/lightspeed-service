"""Functions to adapt OLS configuration for different providers on the go during multi-provider test scenarios."""

import os
import yaml
from tests.e2e.utils import cluster as cluster_utils
from tests.e2e.utils.retry import retry_until_timeout_or_success


def patch_rag_versions() -> None:
    """Patch RAG index versions to use 4.18 to match current RAG content."""
    try:
        print("Patching RAG index versions to use 4.18...")
        
        # Get current OLSConfig
        result = cluster_utils.run_oc([
            "get", "olsconfig", "cluster", "-o", "yaml"
        ])
        
        olsconfig = yaml.safe_load(result.stdout)
        
        # Check if RAG configuration exists
        if "spec" in olsconfig and "ols" in olsconfig["spec"] and "rag" in olsconfig["spec"]["ols"]:
            rag_configs = olsconfig["spec"]["ols"]["rag"]
            
            # Update RAG index paths to use 4.18
            for rag_config in rag_configs:
                if "indexPath" in rag_config:
                    original_path = rag_config["indexPath"]
                    # Replace 4.16 or 4.17 with 4.18 in the path
                    if "/4.16" in original_path:
                        rag_config["indexPath"] = original_path.replace("/4.16", "/4.18")
                        rag_config["indexID"] = rag_config["indexID"].replace("4_16", "4_18")
                        print(f"Updated RAG path from 4.16 to 4.18: {rag_config['indexPath']}")
                    elif "/4.17" in original_path and "/4.18" not in original_path:
                        # Only update 4.17 to 4.18 if 4.18 doesn't already exist
                        has_4_18 = any("/4.18" in config.get("indexPath", "") for config in rag_configs)
                        if not has_4_18:
                            rag_config["indexPath"] = original_path.replace("/4.17", "/4.18")
                            rag_config["indexID"] = rag_config["indexID"].replace("4_17", "4_18")
                            print(f"Updated RAG path from 4.17 to 4.18: {rag_config['indexPath']}")
            
            # Apply the updated configuration
            updated_yaml = yaml.dump(olsconfig)
            cluster_utils.run_oc(["apply", "-f", "-"], command=updated_yaml)
            print("RAG configuration patched successfully")
        else:
            print("No RAG configuration found to patch")
            
    except Exception as e:
        print(f"Warning: Could not patch RAG versions: {e}")
        # Don't fail the entire process for this


def patch_mcp_servers_for_tool_calling(is_tool_calling: bool) -> None:
    """Add MCP servers configuration for tool calling functionality."""
    if not is_tool_calling:
        print("Not a tool calling configuration, skipping MCP server setup")
        return
        
    try:
        print("Adding MCP servers configuration for tool calling...")
        
        # Get current OLSConfig
        result = cluster_utils.run_oc([
            "get", "olsconfig", "cluster", "-o", "yaml"
        ])
        
        olsconfig = yaml.safe_load(result.stdout)
        
        # Add MCP servers configuration if not present
        if "spec" not in olsconfig:
            olsconfig["spec"] = {}
            
        # Define the OpenShift MCP server configuration
        mcp_servers_config = [
            {
                "name": "openshift",
                "transport": "stdio",
                "stdio": {
                    "command": "python",
                    "args": ["./mcp_local/openshift.py"],
                    "env": {}
                }
            }
        ]
        
        # Add or update MCP servers
        olsconfig["spec"]["mcpServers"] = mcp_servers_config
        
        # Apply the updated configuration
        updated_yaml = yaml.dump(olsconfig)
        cluster_utils.run_oc(["apply", "-f", "-"], command=updated_yaml)
        print("MCP servers configuration added successfully")
        print(f"Added MCP servers: {[server['name'] for server in mcp_servers_config]}")
        
    except Exception as e:
        print(f"Warning: Could not add MCP servers configuration: {e}")
        # Don't fail the entire process for this



def adapt_ols_config() -> None:
    """Adapt OLS configuration for different providers dynamically and enable tool calling if requested."""
    print("Adapting OLS configuration for provider switching")

    provider_env = os.getenv("PROVIDER", "openai")
    provider_list = provider_env.split() or ["openai"]
    print(f"Configuring for providers: {provider_list}")

    ols_config_suffix = os.getenv("OLS_CONFIG_SUFFIX", "default")
    tool_calling_enabled = os.getenv("TOOL_CALLING_ENABLED", "n") == "y"

    if tool_calling_enabled and ols_config_suffix == "default":
        ols_config_suffix = "tool_calling"
        print("Using TOOL_CALLING_ENABLED for backward compatibility")

    is_tool_calling = ols_config_suffix == "tool_calling"
    namespace = "openshift-lightspeed"

    try:
        # Choose CRD YAML based on provider config
        if len(provider_list) == 1:
            provider = provider_list[0]
            crd_yml_name = f"olsconfig.crd.{provider}"
            if is_tool_calling:
                crd_yml_name += "_tool_calling"
            print(f"Applying olsconfig CR from {crd_yml_name}.yaml")
            cluster_utils.run_oc(
                ["apply", "-f", f"tests/config/operator_install/{crd_yml_name}.yaml"],
                ignore_existing_resource=True,
            )

            # Configure tool calling if needed
            if is_tool_calling:
                print("Configuring tool calling functionality...")
                
                # 1. Patch introspectionEnabled field
                print("Patching OLSConfig to enable tool calling (spec.ols.introspectionEnabled: true)")
                cluster_utils.run_oc([
                    "patch", "olsconfig", "cluster",
                    "--type=merge",
                    "-p", '{"spec":{"ols":{"introspectionEnabled": true}}}'
                ])

                # 2. Add MCP servers configuration (CRITICAL for tool calling)
                print("Adding MCP servers configuration for tool calling...")
                mcp_servers_patch = {
                    "spec": {
                        "mcpServers": [
                            {
                                "name": "openshift",
                                "transport": "stdio",
                                "stdio": {
                                    "command": "python",
                                    "args": ["./mcp_local/openshift.py"],
                                    "env": {}
                                }
                            }
                        ]
                    }
                }
                
                import json
                cluster_utils.run_oc([
                    "patch", "olsconfig", "cluster",
                    "--type=merge",
                    "-p", json.dumps(mcp_servers_patch)
                ])
                print("✅ Tool calling configuration completed successfully")

        else:
            print("Applying evaluation olsconfig CR for multiple providers")
            cluster_utils.run_oc(
                ["apply", "-f", "tests/config/operator_install/olsconfig.crd.evaluation.yaml"],
                ignore_existing_resource=True,
            )
        print("OLSConfig CR applied successfully")

        # Patch RAG versions to match current content (4.18)
        patch_rag_versions()

    except Exception as e:
        raise RuntimeError(f"Error applying OLSConfig CR: {e}") from e

    # Ensure pod-reader role exists
    try:
        print("Ensuring 'pod-reader' role exists...")
        role_exists = cluster_utils.run_oc([
            "get", "role", "pod-reader", "-n", namespace, "--ignore-not-found"
        ]).stdout.strip()

        if not role_exists:
            print("Creating role 'pod-reader'")
            cluster_utils.run_oc([
                "create", "role", "pod-reader",
                "--verb=get,list", "--resource=pods",
                "--namespace", namespace
            ])
    except Exception as e:
        print(f"Warning: Could not ensure pod-reader role exists: {e}")

    # Ensure test-user service account and rolebinding exist
    try:
        print("Ensuring 'test-user' service account exists...")
        sa_exists = cluster_utils.run_oc([
            "get", "sa", "test-user", "-n", namespace, "--ignore-not-found"
        ]).stdout.strip()

        if not sa_exists:
            print("Creating service account 'test-user'")
            cluster_utils.run_oc(["create", "sa", "test-user", "-n", namespace])

        print("Ensuring 'test-user-pod-reader' role binding exists...")
        rb_exists = cluster_utils.run_oc([
            "get", "rolebinding", "test-user-pod-reader", "-n", namespace, "--ignore-not-found"
        ]).stdout.strip()

        if not rb_exists:
            print("Creating role binding 'test-user-pod-reader'")
            cluster_utils.run_oc([
                "create", "rolebinding", "test-user-pod-reader",
                "--role=pod-reader",
                f"--serviceaccount={namespace}:test-user",
                "--namespace", namespace,
            ])

        print("Service account and role binding verified.")
    except Exception as e:
        raise RuntimeError(f"Error ensuring RBAC setup: {e}") from e

    # Wait for controller manager to detect config change
    print("Waiting for OLS controller to apply updated configuration...")

    deployment_ready = retry_until_timeout_or_success(
        30,
        6,
        lambda: cluster_utils.run_oc([
            "get", "deployment", "lightspeed-app-server",
            "--ignore-not-found", "-o", "name"
        ]).stdout.strip() == "deployment.apps/lightspeed-app-server",
        "Waiting for lightspeed-app-server deployment to be detected",
    )

    if not deployment_ready:
        raise TimeoutError("Timed out waiting for lightspeed-app-server deployment update")

    print("Waiting for pods to be ready after configuration update...")
    cluster_utils.wait_for_running_pod()

    print("OLS configuration and access setup completed successfully.")


if __name__ == "__main__":
    adapt_ols_config()
