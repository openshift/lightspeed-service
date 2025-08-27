"""Functions to adapt OLS configuration for different providers.

Handles multi-provider test scenarios dynamically.
"""

import os

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


def setup_rbac(namespace: str) -> None:
    """Set up pod-reader role and binding.

    Args:
        namespace: The Kubernetes namespace for RBAC configuration.
    """
    print("Ensuring 'pod-reader' role and rolebinding exist...")
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
    print("RBAC setup verified.")


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

    print("Waiting for pods to be ready after configuration update...")
    cluster_utils.wait_for_running_pod()


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
        apply_olsconfig(provider_list)
    except Exception as e:
        raise RuntimeError(f"Error applying OLSConfig CR: {e}") from e

    # Scale controller manager back up to reconcile changes to the olsconfig
    cluster_utils.run_oc(
        [
            "scale",
            "deployment/lightspeed-operator-controller-manager",
            "--replicas",
            "1",
        ]
    )
    retry_until_timeout_or_success(
        30,
        6,
        lambda: cluster_utils.get_pod_by_prefix(
            prefix="lightspeed-operator-controller-manager"
        ),
    )

    wait_for_deployment()

    # scale down the operator controller manager to avoid it interfering with the tests
    cluster_utils.run_oc(
        [
            "scale",
            "deployment/lightspeed-operator-controller-manager",
            "--replicas",
            "0",
        ]
    )
    cluster_utils.run_oc(
        [
            "scale",
            "deployment/lightspeed-app-server",
            "--replicas",
            "0",
        ]
    )

    # Update OLS configmap with additional e2e configurations
    try:
        update_ols_configmap()
    except Exception as e:
        print(f"Warning: Could not update OLS configmap: {e}")
    cluster_utils.run_oc(
        [
            "scale",
            "deployment/lightspeed-app-server",
            "--replicas",
            "1",
        ]
    )

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

    # Wait for deployment and pods
    wait_for_deployment()

    # Disable collector script by default to avoid running during all tests
    pod_name = cluster_utils.get_pod_by_prefix()[0]
    print(f"Disabling collector on pod {pod_name}")
    cluster_utils.create_file(pod_name, OLS_COLLECTOR_DISABLING_FILE, "")

    # Fetch tokens for service accounts
    print("Fetching tokens for service accounts...")
    token = cluster_utils.get_token_for("test-user")
    metrics_token = cluster_utils.get_token_for("metrics-test-user")

    # Set up route and get URL
    ols_url = setup_route()
    wait_for_ols(ols_url)

    print("OLS configuration and access setup completed successfully.")
    return ols_url, token, metrics_token


if __name__ == "__main__":
    adapt_ols_config()
