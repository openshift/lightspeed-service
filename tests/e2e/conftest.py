"""Additional arguments for pytest."""

import pytest
from pytest import TestReport

import tests.e2e.test_api as test_api

makereport_called = False


def pytest_runtest_makereport(item, call) -> TestReport:
    """Generate a synthetic test report.

    If OLS did not come up in time generate a synethic test entry,
    generate a normal test report entry otherwise.
    """
    global makereport_called
    # The first time we try to generate a test report, check if OLS timed out on startup
    # if so, generate a synthetic test report for the wait_for_ols timeout.
    if not test_api.OLS_READY and not makereport_called:
        makereport_called = True
        return TestReport(
            "test_wait_for_ols",
            ["", 0, ""],
            None,
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
    if not test_api.OLS_READY:
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
        default="ibm/granite-13b-chat-v2",
        type=str,
        help="Model for which responses will be evaluated.",
    )
    parser.addoption(
        "--eval_out_dir",
        default="tests/test_results",
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
