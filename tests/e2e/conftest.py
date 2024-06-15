"""Additional arguments for pytest."""


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
