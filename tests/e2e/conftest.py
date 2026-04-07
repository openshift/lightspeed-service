"""Additional arguments for pytest."""

# we add new attributes into pytest instance, which is not recognized
# properly by linters
# pyright: reportAttributeAccessIssue=false

import json
import os

import pytest
from httpx import Client
from pytest import TestReport

from tests.e2e.utils import client as client_utils
from tests.e2e.utils import cluster, ols_installer
from tests.e2e.utils.adapt_ols_config import adapt_ols_config
from tests.e2e.utils.mcp_setup import setup_mcp_on_cluster, teardown_mcp_on_cluster
from tests.e2e.utils.wait_for_ols import wait_for_ols
from tests.scripts.must_gather import must_gather

# this flag is set to True when synthetic test report was already generated
pytest.makereport_called = False

# generic HTTP client for talking to OLS, when OLS is run on a cluster
# this client will be preconfigured with a valid user token header.
pytest.client: Client = None  # pyright: ignore[reportInvalidTypeForm]
pytest.metrics_client: Client = None  # pyright: ignore[reportInvalidTypeForm]
OLS_READY = False
# on_cluster attribute is set to true when the tests are being run
# against ols running on a cluster
on_cluster: bool = False  # pylint: disable=C0103


def _maybe_setup_mcp() -> None:
    """Deploy mock MCP server and secret if running MCP test suite.

    Must run before adapt_ols_config/install_ols so the mock server
    is reachable and the secret exists when the operator reconciles the CR.
    """
    ols_config_suffix = os.getenv("OLS_CONFIG_SUFFIX", "default")
    if "mcp" not in ols_config_suffix:
        return
    print("MCP test suite detected - deploying mock server and secret...")
    setup_mcp_on_cluster()


def pytest_sessionstart():
    """Set up common artifacts used by all e2e tests."""
    global OLS_READY  # pylint: disable=W0603
    global on_cluster  # pylint: disable=W0603
    provider = os.getenv("PROVIDER")
    # OLS_URL env only needs to be set when running against a local ols instance,
    # when ols is run against a cluster the url is retrieved from the cluster.
    ols_url = os.getenv("OLS_URL", "")
    if "localhost" not in ols_url:
        on_cluster = True
        try:
            _maybe_setup_mcp()

            if os.getenv("SKIP_CLUSTER_SETUP", "false").lower() == "true":
                print(
                    "SKIP_CLUSTER_SETUP enabled - skipping OLS installation/configuration."
                )
                print("Using existing cluster setup...")
                cluster.run_oc(
                    ["project", "openshift-lightspeed"], ignore_existing_resource=True
                )
                ols_url = cluster.get_ols_url("ols")
                token = cluster.get_token_for("test-user")
                metrics_token = cluster.get_token_for("metrics-test-user")
                print(f"Using OLS URL: {ols_url}")
            else:
                result = cluster.run_oc(
                    [
                        "get",
                        "clusterserviceversion",
                        "-n",
                        "openshift-lightspeed",
                        "-o",
                        "json",
                    ]
                )
                csv_data = json.loads(result.stdout)
                print(csv_data)

                if not csv_data["items"]:
                    print("OLS Operator is not installed yet.")
                    ols_url, token, metrics_token = ols_installer.install_ols()
                else:
                    print("OLS Operator is already installed. Skipping install.")
                    provider = os.getenv("PROVIDER", "openai")
                    creds = os.getenv("PROVIDER_KEY_PATH", "empty")
                    # create the llm api key secret ols will mount
                    provider_list = provider.split()
                    creds_list = creds.split()
                    for i, prov in enumerate(provider_list):
                        ols_installer.create_secrets(
                            prov, creds_list[i], len(provider_list)
                        )
                    ols_url, token, metrics_token = adapt_ols_config()

        except Exception as e:
            print(f"Error setting up OLS on cluster: {e}")
            must_gather()
            raise e
    else:
        print("Setting up for standalone test execution\n")
        # these variables must be created, but does not have to contain
        # anything relevant for local testing (values are optional)
        token = None
        metrics_token = None

    pytest.client = client_utils.get_http_client(ols_url, token)
    pytest.metrics_client = client_utils.get_http_client(ols_url, metrics_token)
    pytest.ols_url = ols_url

    # Wait for OLS to be ready
    print(f"Waiting for OLS to be ready at url: {ols_url} with provider: {provider}...")
    OLS_READY = wait_for_ols(ols_url)
    print(f"OLS is ready: {OLS_READY}")
    # Gather OLS artifacts in case OLS does not become ready
    if on_cluster and not OLS_READY:
        must_gather()


def pytest_runtest_makereport(item, call) -> TestReport:
    """Generate a synthetic test report.

    If OLS did not come up in time generate a synethic test entry,
    generate a normal test report entry otherwise.
    """
    # The first time we try to generate a test report, check if OLS timed out on startup
    # if so, generate a synthetic test report for the wait_for_ols timeout.
    if not OLS_READY and not pytest.makereport_called:
        pytest.makereport_called = True
        return TestReport(
            "test_wait_for_ols",
            ("", 0, ""),
            {},
            "failed",
            "wait for OLS to startup before running tests",
            "call",
            [],
        )
    # The second time we are called to generate a report, assuming OLS timed out,
    # exit pytest so we don't try to run any more tests (they will fail anyway since
    # OLS didn't come up)
    # There is no clean way to return the synthetic test report above *and* exit pytest
    # in a single invocation
    if not OLS_READY:
        pytest.exit("OLS did not become ready!", 1)
    # If OLS did come up cleanly during setup, then just generate normal test reports for all tests
    return TestReport.from_item_and_call(item, call)


def pytest_addoption(parser):
    """Argument parser for pytest."""
    parser.addoption(
        "--eval_provider",
        default="watsonx",
        type=str,
        help="Provider name, currently used only to form output file name.",
    )
    parser.addoption(
        "--eval_model",
        default="ibm/granite-4-h-small",
        type=str,
        help="Model for which responses will be evaluated.",
    )
    parser.addoption(
        "--eval_provider_model_id",
        nargs="+",
        default=[
            "watsonx+ibm/granite-4-h-small",
            "openai+gpt-4.1-mini",
            "azure_openai+gpt-4.1-mini",
        ],
        type=str,
        help="Identifier for Provider/Model to be used for model eval.",
    )
    parser.addoption(
        "--eval_out_dir",
        default=None,
        type=str,
        help="Result destination.",
    )
    parser.addoption(
        "--eval_query_ids",
        nargs="+",
        default=None,
        help="Ids of questions to be validated. Check json file for valid ids.",
    )
    parser.addoption(
        "--eval_scenario",
        choices=["with_rag", "without_rag"],
        default="with_rag",
        type=str,
        help="Scenario for which responses will be evaluated.",
    )
    parser.addoption(
        "--use_default_json_data",
        default=True,
        type=bool,
        help="When True, default data will be used.",
    )
    parser.addoption(
        "--qna_pool_file",
        default=None,
        type=str,
        help="Additional file having QnA pool in parquet format.",
    )
    parser.addoption(
        "--eval_type",
        choices=["consistency", "model", "all"],
        default="model",
        help="Evaluation type.",
    )
    parser.addoption(
        "--eval_metrics",
        nargs="+",
        default=["cos_score"],
        help="Evaluation score/metric.",
    )
    parser.addoption(
        "--eval_modes",
        nargs="+",
        default=["ols"],
        help="Evaluation modes ex: with just prompt/rag etc.",
    )


def pytest_collection_modifyitems(items: list) -> None:
    """Filter and reorder collected tests.

    - Deselect mcp-marked tests when the MCP suite is not active.
    - Ensure test_user_data_collection runs last in the data_export suite.
    """
    ols_config_suffix = os.getenv("OLS_CONFIG_SUFFIX", "default")
    mcp_enabled = "mcp" in ols_config_suffix

    selected = []
    deselected = []
    for item in items:
        if not mcp_enabled and item.get_closest_marker("mcp"):
            deselected.append(item)
        else:
            selected.append(item)

    if deselected:
        config = items[0].config
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)

    deferred = []
    rest = []
    for item in items:
        if item.name == "test_user_data_collection":
            deferred.append(item)
        else:
            rest.append(item)
    items[:] = rest + deferred


def pytest_sessionfinish():
    """Gather OLS artifacts and clean up test resources after session finishes."""
    if on_cluster:
        ols_config_suffix = os.getenv("OLS_CONFIG_SUFFIX", "default")
        if "mcp" in ols_config_suffix:
            teardown_mcp_on_cluster()
        must_gather()
