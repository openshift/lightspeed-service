"""Additional arguments for pytest."""


def pytest_addoption(parser):
    """Argument parser for pytest."""
    parser.addoption("--eval_model", default="gpt", help="Model to be evaluated.")
    parser.addoption(
        "--eval_threshold", default="0.5", help="Evaluation threshold value."
    )
