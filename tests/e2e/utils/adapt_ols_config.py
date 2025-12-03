"""Functions to adapt OLS configuration for different providers.

Handles multi-provider test scenarios dynamically.
"""

import os
import time

import yaml

from ols.constants import DEFAULT_CONFIGURATION_FILE
from tests.e2e.utils import cluster as cluster_utils
from tests.e2e.utils.data_collector_control import configure_exporter_for_e2e_tests
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

        # Configure user data collection only for data_export test suite
        # Other test suites don't need it and the volume might not be mounted
        ols_config_suffix = os.getenv("OLS_CONFIG_SUFFIX", "default")
        if ols_config_suffix == "data_export":
            olsconfig["ols_config"]["user_data_collection"] = {
                "feedback_disabled": False,
                "feedback_storage": "/app-root/ols-user-data/feedback",
                "transcripts_disabled": False,
                "transcripts_storage": "/app-root/ols-user-data/transcripts",
            }

        # Update the configmap
        configmap["data"][DEFAULT_CONFIGURATION_FILE] = yaml.dump(olsconfig)
        updated_configmap = yaml.dump(configmap)
        cluster_utils.run_oc(["apply", "-f", "-"], command=updated_configmap)
        print("OLS configmap updated successfully")

    except Exception as e:
        raise RuntimeError(
            f"Failed to update OLS configmap with e2e settings: {e}"
        ) from e


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
    print("Waiting for OLS deployment to be available...")
    retry_until_timeout_or_success(
        30,
        5,
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

    print("Waiting for pods to be ready...")
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


def adapt_ols_config() -> tuple[str, str, str]:  # pylint: disable=R0915
    """Adapt OLS configuration for different providers dynamically.

    Ensures RBAC, service accounts, and OLS route exist for test execution.
    This function assumes the operator has already been scaled down during initial setup.

    Returns:
        tuple: (ols_url, token, metrics_token)
    """
    print("Adapting OLS configuration for provider switching")
    provider_env = os.getenv("PROVIDER", "openai")
    provider_list = provider_env.split() or ["openai"]
    ols_image = os.getenv("OLS_IMAGE", "")
    namespace = "openshift-lightspeed"

    print("Checking for existing app server deployment...")
    try:
        cluster_utils.run_oc(
            ["scale", "deployment/lightspeed-app-server", "--replicas", "0"]
        )
        retry_until_timeout_or_success(
            30,
            3,
            lambda: not cluster_utils.get_pod_by_prefix(fail_not_found=False),
            "Waiting for old app server pod to terminate",
        )
        print("Old app server scaled down")
    except Exception as e:
        print(f"No existing app server to scale down (this is OK): {e}")

    try:
        cluster_utils.run_oc(["delete", "olsconfig", "cluster", "--ignore-not-found"])
        print(" Old OLSConfig CR removed")
    except Exception as e:
        print(f"Could not delete old OLSConfig: {e}")

    try:
        apply_olsconfig(provider_list)
        print("New OLSConfig CR applied")
    except Exception as e:
        raise RuntimeError(f"Failed to apply OLSConfig CR: {e}") from e

    cluster_utils.run_oc(
        [
            "scale",
            "deployment/lightspeed-operator-controller-manager",
            "--replicas",
            "1",
        ]
    )
    # Wait for operator
    retry_until_timeout_or_success(
        30,
        5,
        lambda: cluster_utils.get_pod_by_prefix(
            prefix="lightspeed-operator-controller-manager", fail_not_found=False
        ),
        "Waiting for operator to start",
    )

    print("Waiting for operator to reconcile OLSConfig CR (30 seconds)...")
    time.sleep(30)  # Let operator reconcile CR → deployment + configmap

    # Verify reconciliation happened - check deployment exists AND has pods
    print("Verifying operator reconciliation completed...")
    retry_until_timeout_or_success(
        30,  # Give more time for operator to fully reconcile
        3,
        lambda: cluster_utils.run_oc(
            [
                "get",
                "deployment",
                "lightspeed-app-server",
                "--ignore-not-found",
                "-o",
                "jsonpath={.status.replicas}",
            ]
        ).stdout.strip()
        != "",
        "Waiting for operator to create deployment with replicas",
    )
    cluster_utils.run_oc(
        [
            "scale",
            "deployment/lightspeed-operator-controller-manager",
            "--replicas",
            "0",
        ]
    )

    retry_until_timeout_or_success(
        30,
        3,
        lambda: not cluster_utils.get_pod_by_prefix(
            prefix="lightspeed-operator-controller-manager", fail_not_found=False
        ),
        "Waiting for operator to scale down",
    )
    print("Operator scaled down")

    # Scale down app server to apply e2e configurations
    print("Scaling down app server to apply e2e configurations...")
    cluster_utils.run_oc(
        ["scale", "deployment/lightspeed-app-server", "--replicas", "0"]
    )

    retry_until_timeout_or_success(
        30,
        3,
        lambda: not cluster_utils.get_pod_by_prefix(fail_not_found=False),
        "Waiting for app server pod to terminate",
    )
    print("App server scaled down")

    # Update configmap with e2e-specific settings - FAIL FAST if this breaks
    print("Updating configmap with e2e test settings...")
    update_ols_configmap()
    print(" Configmap updated successfully")
    # Apply test image
    if ols_image:
        print(f"Applying test image: {ols_image}")
        try:
            # Patch the lightspeed-service-api container (containers/0)
            patch = (
                f'[{{"op": "replace", "path": "/spec/template/spec/'
                f'containers/0/image", "value": "{ols_image}"}}]'
            )
            cluster_utils.run_oc(
                [
                    "patch",
                    "deployment/lightspeed-app-server",
                    "--type",
                    "json",
                    "-p",
                    patch,
                ]
            )

            print("Image configuration completed")
        except Exception as e:
            print(f" Warning: Could not apply test image: {e}")

    # Scale back up
    print("Scaling up app server with new configuration...")
    cluster_utils.run_oc(
        ["scale", "deployment/lightspeed-app-server", "--replicas", "1"]
    )

    # Wait for deployment to be ready
    wait_for_deployment()

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

    # Configure exporter for e2e tests with proper settings
    try:
        print("Configuring exporter for e2e tests...")
        configure_exporter_for_e2e_tests(
            interval_seconds=3600,  # 1 hour to prevent interference
            ingress_env="stage",
            log_level="DEBUG",
            data_dir="/app-root/ols-user-data",
        )
        print("Exporter configured successfully")
    except Exception as e:
        print(f"Warning: Could not configure exporter: {e}")
        print("Tests may experience interference from data collector")

    # Fetch tokens for service accounts
    print("Fetching tokens for service accounts...")
    token = cluster_utils.get_token_for("test-user")
    metrics_token = cluster_utils.get_token_for("metrics-test-user")

    # Set up route and get URL
    ols_url = setup_route()

    # Wait for OLS to be ready
    print(f"Waiting for OLS to be ready at {ols_url}...")
    if not wait_for_ols(ols_url, timeout=180):
        raise RuntimeError("OLS failed to become ready after configuration")

    print("OLS configuration and access setup completed successfully.")
    return ols_url, token, metrics_token


if __name__ == "__main__":
    adapt_ols_config()
