"""Utilities for deploying the mock MCP server on an OpenShift cluster.

Handles creating the auth secret, deploying the mock server as a pod+service,
and tearing everything down.
The OLS configmap is NOT patched here -- MCP config lives in the OLSConfig CR
(olsconfig.crd.openai_mcp.yaml) and the operator generates the configmap.
"""

from pathlib import Path

from tests.e2e.utils import cluster as cluster_utils
from tests.e2e.utils.retry import retry_until_timeout_or_success

NAMESPACE = "openshift-lightspeed"
MOCK_SERVER_NAME = "mcp-mock-server"
SERVER_DIR = Path(__file__).resolve().parents[1] / "mcp" / "server"
DEPLOYMENT_YAML = SERVER_DIR / "deployment.yaml"

MCP_TOKEN_SECRET_NAME = "mcp-test-token"  # noqa: S105
MCP_TOKEN_VALUE = "Bearer test-secret-token-123"  # noqa: S105


def _deploy_mock_server() -> None:
    """Deploy the mock MCP server pod and service on the cluster."""
    cluster_utils.run_oc(
        ["apply", "-f", str(DEPLOYMENT_YAML)],
        ignore_existing_resource=True,
    )

    retry_until_timeout_or_success(
        60,
        5,
        lambda: bool(
            cluster_utils.get_pod_by_prefix(
                prefix=MOCK_SERVER_NAME, fail_not_found=False
            )
        ),
        "Waiting for mock MCP server pod to be running",
    )
    print("Mock MCP server deployed and running")


def _create_token_secret() -> None:
    """Create the Kubernetes secret referenced by the mock-file-auth MCP server CR."""
    cluster_utils.run_oc(
        ["delete", "secret", MCP_TOKEN_SECRET_NAME, "--ignore-not-found"]
    )

    cluster_utils.run_oc(
        [
            "create",
            "secret",
            "generic",
            MCP_TOKEN_SECRET_NAME,
            f"--from-literal=apitoken={MCP_TOKEN_VALUE}",
        ],
        ignore_existing_resource=True,
    )
    print(f"Created secret '{MCP_TOKEN_SECRET_NAME}'")


def setup_mcp_on_cluster() -> None:
    """Deploy mock MCP server and create auth secret before OLS CR is applied.

    This must run BEFORE adapt_ols_config() so that:
    1. The mock server is reachable when OLS starts and discovers tools.
    2. The token secret exists for the operator to mount into the OLS pod.
    """
    _create_token_secret()
    _deploy_mock_server()
    print("MCP pre-setup complete (secret + mock server)")


def teardown_mcp_on_cluster() -> None:
    """Remove the mock MCP server deployment, service, and secret from the cluster."""
    try:
        cluster_utils.run_oc(
            ["delete", "deployment", MOCK_SERVER_NAME, "--ignore-not-found"]
        )
        cluster_utils.run_oc(
            ["delete", "service", MOCK_SERVER_NAME, "--ignore-not-found"]
        )
        cluster_utils.run_oc(
            ["delete", "secret", MCP_TOKEN_SECRET_NAME, "--ignore-not-found"]
        )
        print("Mock MCP server resources cleaned up")
    except Exception as e:
        print(f"Warning: MCP cleanup failed: {e}")
