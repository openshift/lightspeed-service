"""Functions to adapt OLS configuration for different providers on the go during multi-provider test scenarios."""

import os
import yaml
from tests.e2e.utils import cluster as cluster_utils
from tests.e2e.utils.retry import retry_until_timeout_or_success
from tests.e2e.utils.wait_for_ols import wait_for_ols
from tests.e2e.utils.constants import OLS_COLLECTOR_DISABLING_FILE
from ols.constants import DEFAULT_CONFIGURATION_FILE


def adapt_ols_config() -> tuple[str, str, str]:
    """Adapt OLS configuration for different providers dynamically.

    Ensures RBAC, service accounts, and OLS route exist for test execution.
    Returns:
        tuple: (ols_url, token, metrics_token)
    """
    print("Adapting OLS configuration for provider switching")

    provider_env = os.getenv("PROVIDER", "openai")
    provider_list = provider_env.split() or ["openai"]
    print(f"Configuring for providers: {provider_list}")

    namespace = "openshift-lightspeed"

    # Apply the correct OLSConfig CR
    try:
        if len(provider_list) == 1:
            provider = provider_list[0]
            crd_yml_name = f"olsconfig.crd.{provider}"
            ols_config_suffix = os.getenv("OLS_CONFIG_SUFFIX", "default")
            if ols_config_suffix != "default":
                crd_yml_name += f"_{ols_config_suffix}"
            print(f"Applying olsconfig CR from {crd_yml_name}.yaml")
            cluster_utils.run_oc(
                ["apply", "-f", f"tests/config/operator_install/{crd_yml_name}.yaml"],
                ignore_existing_resource=False,
            )
        else:
            print("Applying evaluation olsconfig CR for multiple providers")
            cluster_utils.run_oc(
                ["apply", "-f", "tests/config/operator_install/olsconfig.crd.evaluation.yaml"],
                ignore_existing_resource=True,
            )
        print("OLSConfig CR applied successfully")
    except Exception as e:
        raise RuntimeError(f"Error applying OLSConfig CR: {e}") from e

    # Ensure service accounts exist
    try:
        print("Ensuring 'test-user' service account exists...")
        cluster_utils.run_oc(["create", "sa", "test-user", "-n", namespace], ignore_existing_resource=True)

        print("Ensuring 'metrics-test-user' service account exists...")
        cluster_utils.run_oc(["create", "sa", "metrics-test-user", "-n", namespace], ignore_existing_resource=True)

        print("Granting access roles to service accounts...")
        cluster_utils.grant_sa_user_access("test-user", "lightspeed-operator-query-access")
        cluster_utils.grant_sa_user_access("metrics-test-user", "lightspeed-operator-ols-metrics-reader")
    except Exception as e:
        raise RuntimeError(f"Error ensuring service accounts or access roles: {e}") from e

    # Ensure pod-reader role and binding exist
    try:
        print("Ensuring 'pod-reader' role and rolebinding exist...")
        cluster_utils.run_oc([
            "create", "role", "pod-reader",
            "--verb=get,list", "--resource=pods",
            "--namespace", namespace
        ], ignore_existing_resource=False)

        cluster_utils.run_oc([
            "create", "rolebinding", "test-user-pod-reader",
            "--role=pod-reader",
            f"--serviceaccount={namespace}:test-user",
            "--namespace", namespace,
        ], ignore_existing_resource=False)

        print("RBAC setup verified.")
    except Exception as e:
        print(f"Warning: Could not ensure pod-reader role/binding: {e}")

    # Wait for OLS deployment and pods to be ready
    print("Waiting for OLS controller to apply updated configuration...")
    retry_until_timeout_or_success(
        30, 6,
        lambda: cluster_utils.run_oc([
            "get", "deployment", "lightspeed-app-server",
            "--ignore-not-found", "-o", "name"
        ]).stdout.strip() == "deployment.apps/lightspeed-app-server",
        "Waiting for lightspeed-app-server deployment to be detected",
    )

    print("Waiting for pods to be ready after configuration update...")
    cluster_utils.wait_for_running_pod()

    # Add MCP server configuration for tool calling
    print("Adding MCP server configuration for tool calling...")
    try:
        add_mcp_server_config()
        # Restart pods to pick up the new configuration
        cluster_utils.run_oc([
            "scale", "deployment/lightspeed-app-server", "--replicas", "0"
        ])
        cluster_utils.run_oc([
            "scale", "deployment/lightspeed-app-server", "--replicas", "1"
        ])
        print("Restarting pods to apply MCP configuration...")
        cluster_utils.wait_for_running_pod()
    except Exception as e:
        print(f"Warning: Could not add MCP server configuration: {e}")

    # Disable collector script by default to avoid running during all tests
    pod_name = cluster_utils.get_pod_by_prefix()[0]
    print(f"Disabling collector on pod {pod_name}")
    cluster_utils.create_file(pod_name, OLS_COLLECTOR_DISABLING_FILE, "")

    # Reuse existing tokens
    print("Fetching tokens for service accounts...")
    token = cluster_utils.get_token_for("test-user")
    metrics_token = cluster_utils.get_token_for("metrics-test-user")

    # Ensure route exists
    try:
        cluster_utils.run_oc(["delete", "route", "ols"], ignore_existing_resource=False)
    except Exception:
        print("No existing route to delete. Continuing...")

    print("Creating route for OLS access")
    cluster_utils.run_oc(["create", "-f", "tests/config/operator_install/route.yaml"], ignore_existing_resource=False)

    url = cluster_utils.run_oc([
        "get", "route", "ols", "-o", "jsonpath='{.spec.host}'"
    ]).stdout.strip("'")

    ols_url = f"https://{url}"
    wait_for_ols(ols_url)

    print("OLS configuration and access setup completed successfully.")
    return ols_url, token, metrics_token


def add_mcp_server_config() -> None:
    """Add MCP server configuration to enable tool calling."""
    # Get the current configmap
    configmap_yaml = cluster_utils.run_oc(["get", "cm/olsconfig", "-o", "yaml"]).stdout
    configmap = yaml.safe_load(configmap_yaml)
    olsconfig = yaml.safe_load(configmap["data"][DEFAULT_CONFIGURATION_FILE])

    # Check if MCP servers already exist
    if "mcp_servers" in olsconfig:
        print("MCP servers already configured, skipping...")
        return

    print("Adding MCP server configuration for OpenShift tools...")
    
    # Add MCP server configuration for OpenShift tools
    # Use absolute path to be more robust across environments
    olsconfig["mcp_servers"] = [
        {
            "name": "openshift",
            "transport": "stdio",
            "stdio": {
                "command": "python",
                "args": ["/app-root/mcp_local/openshift.py"],
                "env": {
                    "PYTHONPATH": "/app-root"
                }
            }
        }
    ]

    # Update the configmap
    configmap["data"][DEFAULT_CONFIGURATION_FILE] = yaml.dump(olsconfig)
    updated_configmap = yaml.dump(configmap)

    print("Applying MCP server configuration...")
    # Apply the updated configmap using replace to avoid conflicts
    cluster_utils.run_oc(["replace", "-f", "-"], command=updated_configmap)
    print("MCP server configuration applied successfully")


if __name__ == "__main__":
    adapt_ols_config()
