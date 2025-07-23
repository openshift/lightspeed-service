"""Functions to adapt OLS configuration for different providers on the go during multi-provider test scenarios."""

import os
from tests.e2e.utils import cluster as cluster_utils
from tests.e2e.utils.retry import retry_until_timeout_or_success


def adapt_ols_config() -> None:
    """Adapt OLS configuration for different providers dynamically.

    Also ensures the test-user service account and role binding exist to avoid test failures.
    """
    print("Adapting OLS configuration for provider switching")

    provider_env = os.getenv("PROVIDER", "openai")
    provider_list = provider_env.split() or ["openai"]
    print(f"Configuring for providers: {provider_list}")

    tool_calling_enabled = os.getenv("TOOL_CALLING_ENABLED", "n") == "y"
    namespace = "openshift-lightspeed"

    try:
        # Choose CRD YAML based on provider config
        if len(provider_list) == 1:
            provider = provider_list[0]
            crd_yml_name = f"olsconfig.crd.{provider}"
            if tool_calling_enabled:
                crd_yml_name += "_tool_calling"
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
