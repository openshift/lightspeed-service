"""Functions to adapt OLS configuration for different providers on the go during multi-provider test scenarios."""

import os
from tests.e2e.utils import cluster as cluster_utils
from tests.e2e.utils.retry import retry_until_timeout_or_success
from tests.e2e.utils.wait_for_ols import wait_for_ols
from tests.e2e.utils import ols_installer


def adapt_ols_config() -> tuple[str, str, str]:
    """Adapt OLS configuration for different providers dynamically.

    Also ensures RBAC, service accounts, and OLS route exist for test execution.
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
                ignore_existing_resource=True,
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

    # Ensure pod-reader role exists
    try:
        print("Ensuring 'pod-reader' role exists...")
        cluster_utils.run_oc([
            "create", "role", "pod-reader",
            "--verb=get,list", "--resource=pods",
            "--namespace", namespace
        ], ignore_existing_resource=True)
    except Exception as e:
        print(f"Warning: Could not ensure pod-reader role: {e}")

    # Ensure test-user and rolebinding exist
    try:
        print("Ensuring 'test-user' service account exists...")
        cluster_utils.run_oc(["create", "sa", "test-user", "-n", namespace], ignore_existing_resource=True)

        print("Ensuring 'test-user-pod-reader' rolebinding exists...")
        cluster_utils.run_oc([
            "create", "rolebinding", "test-user-pod-reader",
            "--role=pod-reader",
            f"--serviceaccount={namespace}:test-user",
            "--namespace", namespace,
        ], ignore_existing_resource=True)

        print("Service account and role binding verified.")
    except Exception as e:
        raise RuntimeError(f"Error ensuring RBAC setup: {e}") from e

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

    # Create test-user + metrics-user tokens and assign roles
    token, metrics_token = ols_installer.create_and_config_sas()

    # Ensure route exists
    try:
        cluster_utils.run_oc(["delete", "route", "ols"])
    except Exception:
        print("No existing route to delete. Continuing...")

    print("Creating route for OLS access")
    cluster_utils.run_oc(["create", "-f", "tests/config/operator_install/route.yaml"])

    url = cluster_utils.run_oc([
        "get", "route", "ols", "-o", "jsonpath='{.spec.host}'"
    ]).stdout.strip("'")

    ols_url = f"https://{url}"
    wait_for_ols(ols_url)

    print("OLS configuration and access setup completed successfully.")
    return ols_url, token, metrics_token


if __name__ == "__main__":
    adapt_ols_config()
