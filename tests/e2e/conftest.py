"""Additional arguments for pytest."""


def pytest_addoption(parser):
    """Argument parser for pytest."""
    parser.addoption(
        "--eval_model",
        default="gpt",
        type=lambda v: "gpt" if "gpt" in v else "granite",
        help="Model to be evaluated.",
    )
