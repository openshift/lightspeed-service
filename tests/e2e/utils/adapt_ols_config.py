"""Functions to adapt OLS configuration for different providers.

Handles multi-provider test scenarios dynamically.
"""

import os
from typing import Literal

from tests.e2e.utils import cluster as cluster_utils
from tests.e2e.utils import cluster_api
from tests.e2e.utils.data_collector_control import configure_exporter_for_e2e_tests
from tests.e2e.utils.olsconfig_cr import apply_e2e_olsconfig
from tests.e2e.utils.retry import retry_until_timeout_or_success
from tests.e2e.utils.wait_for_ols import wait_for_ols

disconnected = os.getenv("DISCONNECTED", "")

_OLS_APP_DEPLOYMENT_WAIT_ATTEMPTS = 60
_OLS_APP_DEPLOYMENT_WAIT_INTERVAL_S = 10
_MCP_SERVER_READY_ATTEMPTS = 60
_MCP_SERVER_READY_INTERVAL_S = 5


def apply_olsconfig(
    provider_list: list[str], operation: Literal["create", "replace_spec"] = "create"
) -> None:
    """Apply the correct OLSConfig CR based on provider configuration.

    Args:
        provider_list: List of provider names to configure.
        operation: Create the CR or replace its spec in place.
    """
    if len(provider_list) == 1:
        provider = provider_list[0]
        crd_yml_name = f"olsconfig.crd.{provider}"
        ols_config_suffix = os.getenv("OLS_CONFIG_SUFFIX", "default")
        if ols_config_suffix != "default":
            crd_yml_name += f"_{ols_config_suffix}"
        crd_yml_file = f"tests/config/operator_install/{crd_yml_name}.yaml"
        print(f"Applying olsconfig CR from {crd_yml_name}.yaml")
        substitute_environment = "rhoai_vllm_lseval" in crd_yml_file
        if substitute_environment and not os.environ.get("KSVC_URL"):
            raise RuntimeError(
                "KSVC_URL environment variable is not set; "
                "required for rhoai_vllm_lseval CR template"
            )
        apply_e2e_olsconfig(
            crd_yml_file,
            operation,
            suffix=ols_config_suffix,
            substitute_environment=substitute_environment,
            ignore_existing_resource=False,
        )
    else:
        ols_config_suffix = os.getenv("OLS_CONFIG_SUFFIX", "default")
        if ols_config_suffix == "default":
            print("Applying evaluation olsconfig CR for multiple providers")
            crd_yml_file = "tests/config/operator_install/olsconfig.crd.evaluation.yaml"
        else:
            print(
                "Applying multi-provider olsconfig CR for suffix "
                f"{ols_config_suffix}"
            )
            crd_yml_file = (
                "tests/config/operator_install/"
                f"olsconfig.crd.{ols_config_suffix}.yaml"
            )
        apply_e2e_olsconfig(
            crd_yml_file,
            operation,
            suffix=ols_config_suffix,
            substitute_environment=False,
            ignore_existing_resource=False,
        )
    print("OLSConfig CR applied successfully")


def setup_service_accounts(namespace: str) -> None:
    """Set up service accounts and access roles.

    Args:
        namespace: The Kubernetes namespace to create service accounts in.
    """
    print("Ensuring 'test-user' service account exists...")
    cluster_api.create_resource(
        {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {"name": "test-user"},
        },
        namespace,
        ignore_existing=True,
    )

    print("Ensuring 'metrics-test-user' service account exists...")
    cluster_api.create_resource(
        {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {"name": "metrics-test-user"},
        },
        namespace,
        ignore_existing=True,
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
    cluster_api.configure_pod_reader(namespace)
    print("RBAC setup verified.")


def setup_route() -> str:
    """Set up route and return OLS URL.

    Returns:
        The HTTPS URL for accessing the OLS service.
    """
    cluster_api.apply_file("tests/config/operator_install/route.yaml")
    return cluster_utils.get_ols_url("ols")


def _openshift_mcp_server_is_enabled() -> bool:
    """Return whether the reconciled OLSConfig enables the built-in MCP server."""
    result = cluster_api.read_resource("olsconfig", "cluster")
    return result.get("spec", {}).get("ols", {}).get("introspectionEnabled", True)


def _openshift_mcp_server_is_available() -> bool:
    """Return whether the OpenShift MCP server deployment is available."""
    deployment = cluster_api.read_resource("deployment", "openshift-mcp-server")
    status = deployment.get("status", {})
    return status.get("observedGeneration") == deployment["metadata"][
        "generation"
    ] and status.get("availableReplicas", 0) >= deployment["spec"].get("replicas", 1)


def _wait_for_openshift_mcp_server() -> None:
    """Wait for the built-in MCP server when introspection is enabled."""
    if not _openshift_mcp_server_is_enabled():
        return

    if not retry_until_timeout_or_success(
        _MCP_SERVER_READY_ATTEMPTS,
        _MCP_SERVER_READY_INTERVAL_S,
        lambda: cluster_utils.deployment_exists("openshift-mcp-server"),
        "Waiting for openshift-mcp-server deployment to be created",
    ):
        raise RuntimeError("Timed out waiting for openshift-mcp-server Deployment")

    if not retry_until_timeout_or_success(
        _MCP_SERVER_READY_ATTEMPTS,
        _MCP_SERVER_READY_INTERVAL_S,
        _openshift_mcp_server_is_available,
        "Waiting for openshift-mcp-server deployment to become available",
    ):
        raise RuntimeError(
            "Timed out waiting for openshift-mcp-server Deployment to become available"
        )


def _olsconfig_api_is_ready() -> bool:
    """Require readiness for the current CR generation, not the previous suite."""
    config = cluster_api.read_resource("olsconfig", "cluster")
    generation = config["metadata"]["generation"]
    return any(
        condition.get("type") == "ApiReady"
        and condition.get("status") == "True"
        and condition.get("observedGeneration") == generation
        for condition in config.get("status", {}).get("conditions", [])
    )


def _app_server_rollout_complete() -> bool:
    """Require all serving replicas to be from the current pod template."""
    deployment = cluster_api.read_resource("deployment", "lightspeed-app-server")
    status = deployment.get("status", {})
    replicas = deployment["spec"].get("replicas", 1)
    return (
        status.get("observedGeneration") == deployment["metadata"]["generation"]
        and status.get("updatedReplicas") == replicas
        and status.get("availableReplicas") == replicas
        and status.get("replicas") == replicas
    )


def wait_for_olsconfig() -> None:
    """Wait for the current suite spec and app-server rollout."""
    if not retry_until_timeout_or_success(
        _OLS_APP_DEPLOYMENT_WAIT_ATTEMPTS,
        _OLS_APP_DEPLOYMENT_WAIT_INTERVAL_S,
        _olsconfig_api_is_ready,
        "Waiting for the operator to reconcile the current OLSConfig generation",
    ):
        raise RuntimeError("Timed out waiting for OLSConfig ApiReady")
    if not retry_until_timeout_or_success(
        _OLS_APP_DEPLOYMENT_WAIT_ATTEMPTS,
        _OLS_APP_DEPLOYMENT_WAIT_INTERVAL_S,
        _app_server_rollout_complete,
        "Waiting for the app-server Deployment rollout to finish",
    ):
        raise RuntimeError("Timed out waiting for the app-server Deployment rollout")
    if image := os.getenv("OLS_IMAGE"):
        deployment = cluster_api.read_resource("deployment", "lightspeed-app-server")
        containers = deployment["spec"]["template"]["spec"]["containers"]
        actual = next(
            (c["image"] for c in containers if c["name"] == "lightspeed-service-api"),
            None,
        )
        if actual != image:
            raise RuntimeError(
                f"Candidate service image mismatch: expected {image}, got {actual}"
            )
    _wait_for_openshift_mcp_server()


def _reconcile_olsconfig_with_operator(provider_list: list[str]) -> None:
    """Replace the suite spec and let the operator reconcile it."""
    apply_olsconfig(provider_list, operation="replace_spec")
    wait_for_olsconfig()


def switch_default_provider(provider: str, model: str) -> None:
    """Apply matrix defaults through the CR, leaving the operator running."""
    ols = cluster_api.read_resource("olsconfig", "cluster")["spec"]["ols"]
    if (ols.get("defaultProvider"), ols.get("defaultModel")) != (provider, model):
        cluster_api.patch_resource(
            "olsconfig",
            "cluster",
            {"spec": {"ols": {"defaultProvider": provider, "defaultModel": model}}},
        )
        wait_for_olsconfig()


def _setup_access_and_tokens(namespace: str) -> tuple[str, str]:
    """Set up service accounts, RBAC, and return tokens.

    Args:
        namespace: The Kubernetes namespace for access configuration.

    Returns:
        tuple: (token, metrics_token)
    """
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

    if not disconnected:
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

    return token, metrics_token


def adapt_ols_config() -> tuple[str, str, str]:
    """Adapt OLS configuration for different providers dynamically.

    Ensures RBAC, service accounts, and OLS route exist for test execution.
    The caller prepares provider credentials before applying the suite fixture.

    Returns:
        tuple: (ols_url, token, metrics_token)
    """
    print("Adapting OLS configuration for provider switching")
    provider_env = os.getenv("PROVIDER", "openai")
    provider_list = provider_env.split() or ["openai"]
    namespace = "openshift-lightspeed"

    _reconcile_olsconfig_with_operator(provider_list)
    token, metrics_token = _setup_access_and_tokens(namespace)

    # Set up route and get URL
    ols_url = setup_route()

    # Wait for OLS to be ready
    print(f"Waiting for OLS to be ready at {ols_url}...")
    if not wait_for_ols(ols_url, timeout=360):
        raise RuntimeError("OLS failed to become ready after configuration")

    print("OLS configuration and access setup completed successfully.")
    return ols_url, token, metrics_token


if __name__ == "__main__":
    adapt_ols_config()
