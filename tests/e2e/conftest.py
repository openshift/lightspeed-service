"""Additional arguments for pytest."""


def pytest_addoption(parser):
    """Argument parser for pytest."""
    parser.addoption(
        "--eval_model",
        default="gpt",
        type=str,
        help="Model to be evaluated.",
    )
