"""Functions to adapt OLS configuration for different providers.

Handles multi-provider test scenarios dynamically.
"""

import os
import time

import yaml

from ols.constants import DEFAULT_CONFIGURATION_FILE
from tests.e2e.utils import cluster as cluster_utils
from tests.e2e.utils.constants import OLS_COLLECTOR_DISABLING_FILE
from tests.e2e.utils.retry import retry_until_timeout_or_success
from tests.e2e.utils.wait_for_ols import wait_for_ols


def apply_olsconfig(provider_list: list[str]) -> None:
    """Apply the correct OLSConfig CR based on provider configuration.

    Args:
        provider_list: List of provider names to configure.
    """
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
            [
                "apply",
                "-f",
                "tests/config/operator_install/olsconfig.crd.evaluation.yaml",
            ],
            ignore_existing_resource=True,
        )
    print("OLSConfig CR applied successfully")


def update_ols_configmap() -> None:
    """Update OLS configmap with additional e2e test configurations.

    Configures logging levels and user data collector settings for testing.
    """
    try:
        print("Updating OLS configmap for e2e tests...")
        # Get the current configmap
        configmap_yaml = cluster_utils.run_oc(
            ["get", "cm/olsconfig", "-o", "yaml"]
        ).stdout
        configmap = yaml.safe_load(configmap_yaml)
        olsconfig = yaml.safe_load(configmap["data"][DEFAULT_CONFIGURATION_FILE])

        # Ensure proper logging config for e2e tests
        if "ols_config" not in olsconfig:
            olsconfig["ols_config"] = {}
        if "logging_config" not in olsconfig["ols_config"]:
            olsconfig["ols_config"]["logging_config"] = {}

        # Set INFO level to avoid redacted logs
        olsconfig["ols_config"]["logging_config"]["lib_log_level"] = "INFO"

        # Add user data collector config for e2e tests
        olsconfig["user_data_collector_config"] = {
            "data_storage": "/app-root/ols-user-data",
            "log_level": "debug",
            "collection_interval": 10,
            "run_without_initial_wait": True,
            "ingress_env": "stage",
            "cp_offline_token": os.getenv("CP_OFFLINE_TOKEN", ""),
        }

        # Note: MCP is enabled via introspectionEnabled=true in CRD, not configmap
        # The operator automatically provisions MCP sidecar when introspection is enabled

        # Update the configmap
        configmap["data"][DEFAULT_CONFIGURATION_FILE] = yaml.dump(olsconfig)
        updated_configmap = yaml.dump(configmap)
        cluster_utils.run_oc(["apply", "-f", "-"], command=updated_configmap)
        print("OLS configmap updated successfully")

    except Exception as e:
        print(f"Warning: Could not update OLS configmap: {e}")


def setup_service_accounts(namespace: str) -> None:
    """Set up service accounts and access roles.

    Args:
        namespace: The Kubernetes namespace to create service accounts in.
    """
    print("Ensuring 'test-user' service account exists...")
    cluster_utils.run_oc(
        ["create", "sa", "test-user", "-n", namespace],
        ignore_existing_resource=True,
    )

    print("Ensuring 'metrics-test-user' service account exists...")
    cluster_utils.run_oc(
        ["create", "sa", "metrics-test-user", "-n", namespace],
        ignore_existing_resource=True,
    )

    print("Granting access roles to service accounts...")
    cluster_utils.grant_sa_user_access("test-user", "lightspeed-operator-query-access")
    cluster_utils.grant_sa_user_access(
        "metrics-test-user", "lightspeed-operator-ols-metrics-reader"
    )

    # Set up additional permissions for user data collection
    print("Setting up user data collection authentication...")
    try:
        # Grant service accounts access to pull secrets (needed for data collection auth)
        cluster_utils.run_oc(
            [
                "create",
                "clusterrole",
                "pull-secret-reader",
                "--verb=get",
                "--resource=secrets",
                "--resource-name=pull-secret",
            ],
            ignore_existing_resource=True,
        )

        # Also allow reading cluster info (needed for cluster ID)
        cluster_utils.run_oc(
            [
                "create",
                "clusterrole",
                "cluster-info-reader",
                "--verb=get",
                "--resource=namespaces",
                "--resource-name=kube-system",
            ],
            ignore_existing_resource=True,
        )

        # Bind roles to lightspeed service account (used by data collector sidecar)
        cluster_utils.run_oc(
            [
                "create",
                "clusterrolebinding",
                "lightspeed-pull-secret-access",
                "--clusterrole=pull-secret-reader",
                f"--serviceaccount={namespace}:lightspeed-app-server",
            ],
            ignore_existing_resource=True,
        )

        cluster_utils.run_oc(
            [
                "create",
                "clusterrolebinding",
                "lightspeed-cluster-info-access",
                "--clusterrole=cluster-info-reader",
                f"--serviceaccount={namespace}:lightspeed-app-server",
            ],
            ignore_existing_resource=True,
        )

        print("User data collection authentication configured.")
    except Exception as e:
        print(f"Warning: Could not fully configure user data collection auth: {e}")


def setup_rbac(namespace: str) -> None:
    """Set up RBAC roles and bindings for E2E testing.

    Args:
        namespace: The Kubernetes namespace for RBAC configuration.
    """
    print("Setting up RBAC permissions for E2E testing...")

    # Basic pod-reader role for standard tests
    cluster_utils.run_oc(
        [
            "create",
            "role",
            "pod-reader",
            "--verb=get,list",
            "--resource=pods",
            "--namespace",
            namespace,
        ],
        ignore_existing_resource=True,
    )

    cluster_utils.run_oc(
        [
            "create",
            "rolebinding",
            "test-user-pod-reader",
            "--role=pod-reader",
            f"--serviceaccount={namespace}:test-user",
            "--namespace",
            namespace,
        ],
        ignore_existing_resource=True,
    )

    # Additional permissions for user data collection and MCP sidecar
    print("Setting up additional permissions for sidecar containers...")

    # Cluster-wide permissions needed for MCP sidecar and data collection
    cluster_utils.run_oc(
        [
            "create",
            "clusterrole",
            "ols-sidecar-permissions",
            "--verb=get,list,watch",
            "--resource=nodes,namespaces,pods,services,secrets,configmaps",
        ],
        ignore_existing_resource=True,
    )

    # Bind to the lightspeed service account (used by sidecars)
    cluster_utils.run_oc(
        [
            "create",
            "clusterrolebinding",
            "ols-sidecar-binding",
            "--clusterrole=ols-sidecar-permissions",
            f"--serviceaccount={namespace}:lightspeed-app-server",
        ],
        ignore_existing_resource=True,
    )

    # Also bind to test-user for testing scenarios
    cluster_utils.run_oc(
        [
            "create",
            "clusterrolebinding",
            "test-user-sidecar-binding",
            "--clusterrole=ols-sidecar-permissions",
            f"--serviceaccount={namespace}:test-user",
        ],
        ignore_existing_resource=True,
    )

    print("RBAC setup completed with sidecar permissions.")


def check_pod_startup_issues() -> None:
    """Check for common issues that prevent pods from starting."""
    print("Checking for common pod startup issues...")

    try:
        # Check if there are any pods in pending state
        pending_pods = cluster_utils.run_oc(
            [
                "get",
                "pods",
                "-o",
                "jsonpath={.items[?(@.status.phase=='Pending')].metadata.name}",
            ]
        ).stdout.strip()

        if pending_pods:
            print(f"⚠️  Found pending pods: {pending_pods}")

            # Check events for these pods
            for pod_name in pending_pods.split():
                if pod_name:
                    print(f"Checking events for pod: {pod_name}")
                    pod_events = cluster_utils.run_oc(
                        [
                            "get",
                            "events",
                            "--field-selector",
                            f"involvedObject.name={pod_name}",
                        ]
                    ).stdout
                    print(f"Events for {pod_name}:")
                    print(pod_events)

        # Check for image pull issues
        image_pull_errors = cluster_utils.run_oc(
            [
                "get",
                "events",
                "--field-selector",
                "reason=Failed",
                "-o",
                "jsonpath={.items[?(@.reason=='Failed')].message}",
            ]
        ).stdout

        if "image" in image_pull_errors.lower() or "pull" in image_pull_errors.lower():
            print("⚠️  Potential image pull issues detected")
            print(image_pull_errors)

    except Exception as e:
        print(f"Could not check for startup issues: {e}")


def wait_for_deployment() -> None:
    """Wait for OLS deployment and pods to be ready.

    Ensures the lightspeed-app-server deployment is available and pods are running.
    """
    print("Waiting for OLS controller to apply updated configuration...")
    retry_until_timeout_or_success(
        30,
        6,
        lambda: cluster_utils.run_oc(
            [
                "get",
                "deployment",
                "lightspeed-app-server",
                "--ignore-not-found",
                "-o",
                "name",
            ]
        ).stdout.strip()
        == "deployment.apps/lightspeed-app-server",
        "Waiting for lightspeed-app-server deployment to be detected",
    )

    # Enhanced pod readiness check with MCP sidecar support
    print("Waiting for deployment to be ready...")

    # Check if this is a tool calling configuration (which needs MCP sidecar)
    ols_config_suffix = os.getenv("OLS_CONFIG_SUFFIX", "default")
    is_tool_calling = "tool_calling" in ols_config_suffix

    if is_tool_calling:
        print(
            "Tool calling detected - expecting 3 containers (main + data-collector + mcp-sidecar)"
        )
    else:
        print("Standard configuration - expecting 2 containers (main + data-collector)")

    print("Waiting for pods to be ready after configuration update...")
    try:
        cluster_utils.wait_for_running_pod()

        # For tool calling, give MCP sidecar extra time to initialize
        if is_tool_calling:
            print("Giving MCP sidecar additional time to initialize...")
            time.sleep(15)  # Extra wait for MCP sidecar to be fully ready

        print("✅ Pod containers are ready")

    except Exception as e:
        print(f"❌ Error waiting for pod readiness: {e}")

        # Check for common startup issues
        check_pod_startup_issues()

        # Get debug information
        try:
            pods = cluster_utils.run_oc(["get", "pods", "-o", "wide"]).stdout
            print("Current pods status:")
            print(pods)

            # Get container statuses if pod exists
            try:
                pod_name = cluster_utils.get_pod_by_prefix()[0]
                container_statuses = cluster_utils.run_oc(
                    [
                        "get",
                        "pod",
                        pod_name,
                        "-o",
                        "jsonpath={.status.containerStatuses[*].name}",
                    ]
                ).stdout
                print(f"Container names: {container_statuses}")

                ready_statuses = cluster_utils.run_oc(
                    [
                        "get",
                        "pod",
                        pod_name,
                        "-o",
                        "jsonpath={.status.containerStatuses[*].ready}",
                    ]
                ).stdout
                print(f"Container ready statuses: {ready_statuses}")
            except Exception:
                print("No app server pod found yet")

        except Exception as debug_e:
            print(f"Could not get debug info: {debug_e}")
        raise


def setup_route() -> str:
    """Set up route and return OLS URL.

    Returns:
        The HTTPS URL for accessing the OLS service.
    """
    try:
        cluster_utils.run_oc(["delete", "route", "ols"], ignore_existing_resource=False)
    except Exception:
        print("No existing route to delete. Continuing...")

    print("Creating route for OLS access")
    cluster_utils.run_oc(
        ["create", "-f", "tests/config/operator_install/route.yaml"],
        ignore_existing_resource=False,
    )

    url = cluster_utils.run_oc(
        ["get", "route", "ols", "-o", "jsonpath='{.spec.host}'"]
    ).stdout.strip("'")

    return f"https://{url}"


# MCP validation removed - operator handles MCP sidecar automatically via introspectionEnabled


def _setup_controller_manager(provider_list: list[str]) -> None:
    """Set up controller manager for configuration changes."""
    # Apply the correct OLSConfig CR
    try:
        apply_olsconfig(provider_list)
    except Exception as e:
        raise RuntimeError(f"Error applying OLSConfig CR: {e}") from e

    # Scale controller manager back up to reconcile changes to the olsconfig
    print("Scaling controller manager up to apply new configuration...")
    cluster_utils.run_oc(
        [
            "scale",
            "deployment/lightspeed-operator-controller-manager",
            "--replicas",
            "1",
        ]
    )

    # Wait for controller manager to be ready
    retry_until_timeout_or_success(
        30,
        6,
        lambda: cluster_utils.get_pod_by_prefix(
            prefix="lightspeed-operator-controller-manager"
        ),
    )


def _finalize_deployment_setup() -> None:
    """Finalize deployment setup and configuration."""
    # Wait for the new configuration to be applied and pods to be ready
    print("Waiting for new configuration to be applied...")
    wait_for_deployment()

    # Update OLS configmap with additional e2e configurations
    print("Updating OLS configmap with e2e test configurations...")
    try:
        update_ols_configmap()
    except Exception as e:
        print(f"Warning: Could not update OLS configmap: {e}")

    # Wait a bit more for configmap changes to be picked up
    print("Waiting for configmap changes to be applied...")
    time.sleep(10)

    # Scale down the operator controller manager to avoid it interfering with the tests
    print("Scaling down controller manager to prevent interference...")
    cluster_utils.run_oc(
        [
            "scale",
            "deployment/lightspeed-operator-controller-manager",
            "--replicas",
            "0",
        ]
    )

    # Give the operator time to gracefully shut down
    time.sleep(5)


def _setup_test_environment(namespace: str) -> None:
    """Set up test environment with service accounts and RBAC."""
    # Ensure service accounts exist
    try:
        setup_service_accounts(namespace)
    except Exception as e:
        raise RuntimeError(
            f"Error ensuring service accounts or access roles: {e}"
        ) from e

    # Ensure pod-reader role and binding exist
    try:
        setup_rbac(namespace)
    except Exception as e:
        print(f"Warning: Could not ensure pod-reader role/binding: {e}")


def _wait_for_final_deployment() -> None:
    """Wait for final deployment readiness with enhanced error handling."""
    print("Waiting for deployment to be ready after all configuration changes...")
    try:
        wait_for_deployment()
    except Exception as e:
        print(f"❌ Deployment failed to become ready: {e}")
        # Get detailed pod status for debugging
        try:
            pods = cluster_utils.run_oc(["get", "pods", "-o", "wide"]).stdout
            print("Current pod status:")
            print(pods)

            # Check for any events that might explain the pending state
            events = cluster_utils.run_oc(
                ["get", "events", "--sort-by=.lastTimestamp"]
            ).stdout
            print("Recent events (last 10):")
            print("\n".join(events.split("\n")[-10:]))

        except Exception as debug_e:
            print(f"Could not get debug info: {debug_e}")
        raise


def _finalize_test_setup() -> tuple[str, str]:
    """Finalize test setup and return tokens."""
    # Disable collector script by default to avoid running during all tests
    pod_name = cluster_utils.get_pod_by_prefix()[0]
    print(f"Disabling collector on pod {pod_name}")
    cluster_utils.create_file(pod_name, OLS_COLLECTOR_DISABLING_FILE, "")

    # Fetch tokens for service accounts
    print("Fetching tokens for service accounts...")
    token = cluster_utils.get_token_for("test-user")
    metrics_token = cluster_utils.get_token_for("metrics-test-user")

    return token, metrics_token


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

    # Set up controller manager
    _setup_controller_manager(provider_list)

    # Finalize deployment setup
    _finalize_deployment_setup()

    # Set up test environment
    _setup_test_environment(namespace)

    # Wait for final deployment
    _wait_for_final_deployment()

    # Finalize test setup
    token, metrics_token = _finalize_test_setup()

    # Set up route and get URL
    ols_url = setup_route()
    wait_for_ols(ols_url)

    # MCP sidecar is automatically provisioned by operator when introspectionEnabled: true

    print("OLS configuration and access setup completed successfully.")
    return ols_url, token, metrics_token


if __name__ == "__main__":
    adapt_ols_config()
