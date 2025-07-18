"""Functions to adapt OLS configuration for different providers on the go during multi-provider test scenarios."""

import os
from tests.e2e.utils import cluster as cluster_utils
from tests.e2e.utils.retry import retry_until_timeout_or_success


def get_olsconfig_cr_file(providers: list[str], tool_calling_enabled: bool) -> str:
    """Get the appropriate olsconfig CR file based on providers and options.

    Args:
        providers: List of provider names
        tool_calling_enabled: Whether tool calling is enabled

    Returns:
        Path to the appropriate olsconfig CR file
    """
    # For multiple providers, use evaluation config
    if len(providers) > 1:
        return "tests/config/operator_install/olsconfig.crd.evaluation.yaml"

    # For single provider, select based on provider type
    provider = providers[0]
    base_name = f"olsconfig.crd.{provider}"

    # Check if tool calling is enabled and CR file exists
    if tool_calling_enabled:
        tool_calling_file = f"tests/config/operator_install/{base_name}_tool_calling.yaml"
        if os.path.exists(tool_calling_file):
            base_name += "_tool_calling"

    # Validate that the CR file exists
    cr_file_path = f"tests/config/operator_install/{base_name}.yaml"
    if not os.path.exists(cr_file_path):
        raise FileNotFoundError(f"CR file not found: {base_name}.yaml")

    return cr_file_path


def adapt_ols_config() -> None:
    """Adapt OLS configuration for different providers dynamically.

    This function selects and applies the appropriate olsconfig Custom Resource
    based on environment variables, allowing switching across providers without reinstalling the operator.

    Environment variables:
    - PROVIDER: Space-separated list of provider names.
    - TOOL_CALLING_ENABLED: Set to "y" to enable tool calling (if supported by provider)
    """
    print("Adapting OLS configuration for provider switching")

    # Get provider configuration from environment
    provider_env = os.getenv("PROVIDER", "openai")
    providers = [p for p in provider_env.split() if p.strip()]
    if not providers:
        providers = ["openai"]

    print(f"Configuring for providers: {providers}")

    # Check if tool calling is enabled
    tool_calling_enabled = os.getenv("TOOL_CALLING_ENABLED", "n") == "y"

    # Select appropriate olsconfig CR file
    cr_file = get_olsconfig_cr_file(providers, tool_calling_enabled)
    print(f"Selected olsconfig CR file: {cr_file}")

    # Apply the olsconfig CR
    print(f"Applying olsconfig CR from {cr_file}")
    cluster_utils.run_oc(["apply", "-f", cr_file])
    print("Successfully applied olsconfig CR")

    # Wait for the controller manager to detect the changes and restart the deployment
    print("Waiting for controller manager to detect olsconfig changes and restart deployment")

    # Wait for the deployment to be updated/restarted by the controller manager
    print("Waiting for OLS deployment to be updated")
    deployment_ready = retry_until_timeout_or_success(
        30,
        6,
        lambda: cluster_utils.run_oc([
            "get", "deployment", "lightspeed-app-server",
            "--ignore-not-found", "-o", "name"
        ]).stdout.strip() == "deployment.apps/lightspeed-app-server",
        "Waiting for OLS API server deployment to be updated by controller",
    )

    if not deployment_ready:
        raise Exception("Timed out waiting for OLS deployment to be updated")

    # Wait for the pod to be ready (this handles the initialization phase automatically)
    print("Waiting for OLS pod to be ready after configuration change")
    cluster_utils.wait_for_running_pod()

    print("Configuration adaptation completed successfully")

if __name__ == "__main__":
    adapt_ols_config()
