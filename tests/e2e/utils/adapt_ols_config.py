"""Functions to adapt OLS configuration for different providers.

Handles multi-provider test scenarios dynamically.
"""

import os
import time

import yaml

from ols.constants import DEFAULT_CONFIGURATION_FILE
from tests.e2e.utils import client as client_utils
from tests.e2e.utils import cluster as cluster_utils
from tests.e2e.utils.constants import OLS_SERVICE_DEPLOYMENT
from tests.e2e.utils.data_collector_control import configure_exporter_for_e2e_tests
from tests.e2e.utils.ols_installer import (
    create_secrets,
    get_service_account_tokens,
    setup_rbac,
    setup_route,
    setup_service_accounts,
    update_lcore_setting,
    update_ols_config,
)
from tests.e2e.utils.retry import retry_until_timeout_or_success


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
    This is a wrapper around update_ols_config that adds data_export specific settings.
    """
    # First apply the standard config updates
    update_ols_config()

    # Then add data_export specific user data collection config if needed
    ols_config_suffix = os.getenv("OLS_CONFIG_SUFFIX", "default")
    if ols_config_suffix == "data_export":
        try:
            configmap_yaml = cluster_utils.run_oc(
                ["get", "cm/olsconfig", "-o", "yaml"]
            ).stdout
            configmap = yaml.safe_load(configmap_yaml)
            olsconfig = yaml.safe_load(configmap["data"][DEFAULT_CONFIGURATION_FILE])

            olsconfig["ols_config"]["user_data_collection"] = {
                "feedback_disabled": False,
                "feedback_storage": "/app-root/ols-user-data/feedback",
                "transcripts_disabled": False,
                "transcripts_storage": "/app-root/ols-user-data/transcripts",
            }

            configmap["data"][DEFAULT_CONFIGURATION_FILE] = yaml.dump(olsconfig)
            updated_configmap = yaml.dump(configmap)
            cluster_utils.run_oc(["apply", "-f", "-"], command=updated_configmap)
            print("Data export configmap settings applied successfully")
        except Exception as e:
            raise RuntimeError(
                f"Failed to update OLS configmap with data export settings: {e}"
            ) from e


def wait_for_deployment() -> None:
    """Wait for OLS deployment and pods to be ready.

    Ensures the service deployment is available and pods are running.
    """
    print("Waiting for OLS deployment to be available...")
    retry_until_timeout_or_success(
        30,
        5,
        lambda: cluster_utils.run_oc(
            [
                "get",
                "deployment",
                OLS_SERVICE_DEPLOYMENT,
                "--ignore-not-found",
                "-o",
                "name",
            ]
        ).stdout.strip()
        == f"deployment.apps/{OLS_SERVICE_DEPLOYMENT}",
        "Waiting for lightspeed-app-server deployment to be detected",
    )

    print("Waiting for pods to be ready...")
    cluster_utils.wait_for_running_pod(name=OLS_SERVICE_DEPLOYMENT)


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
    creds = os.getenv("PROVIDER_KEY_PATH", "empty")
    cluster_utils.run_oc(
        ["project", "openshift-lightspeed"], ignore_existing_resource=True
    )

    # Update lcore setting if LCORE is enabled
    update_lcore_setting()
    # Scaling operator to 1 replica to allow finalizer to run for olsconfig
    cluster_utils.run_oc(
        [
            "scale",
            "deployment/lightspeed-operator-controller-manager",
            "--replicas",
            "1",
        ]
    )
    # Wait for operator pod to be ready
    cluster_utils.wait_for_running_pod("lightspeed-operator-controller-manager")
    try:
        cluster_utils.run_oc(["delete", "secret", "llmcreds", "--ignore-not-found"])
    except Exception as e:
        print(f"Could not delete old secret: {e}")
    creds_list = creds.split()
    for i, prov in enumerate(provider_list):
        create_secrets(prov, creds_list[i], len(provider_list))
    try:
        apply_olsconfig(provider_list)
        print("New OLSConfig CR applied")
    except Exception as e:
        raise RuntimeError(f"Failed to apply OLSConfig CR: {e}") from e

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
                OLS_SERVICE_DEPLOYMENT,
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
            "1",
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
        ["scale", f"deployment/{OLS_SERVICE_DEPLOYMENT}", "--replicas", "0"]
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
    if OLS_SERVICE_DEPLOYMENT == "lightspeed-app-server":
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
                    f"deployment/{OLS_SERVICE_DEPLOYMENT}",
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
        ["scale", f"deployment/{OLS_SERVICE_DEPLOYMENT}", "--replicas", "1"]
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

    # Fetch tokens for service accounts
    token, metrics_token = get_service_account_tokens()

    # Set up route and get URL
    ols_url = setup_route()

    # Configure exporter for e2e tests with proper settings
    try:
        print("Configuring exporter for e2e tests...")
        # Create client for the exporter configuration
        test_client = client_utils.get_http_client(ols_url, token)
        configure_exporter_for_e2e_tests(
            client=test_client,
            interval_seconds=3600,  # 1 hour to prevent interference
            ingress_env="stage",
            log_level="DEBUG",
            data_dir="/app-root/ols-user-data",
        )
        print("Exporter configured successfully")
    except Exception as e:
        print(f"Warning: Could not configure exporter: {e}")
        print("Tests may experience interference from data collector")

    print("OLS configuration and access setup completed successfully.")
    return ols_url, token, metrics_token


if __name__ == "__main__":
    adapt_ols_config()
