"""Question validators, statament classifiers, and response generators."""

import re

from ols.utils import config


def camel_to_snake(string: str) -> str:
    """Convert a string from camel case to snake case.

    Args:
        string: The string to be converted.

    Returns:
        The converted string.
    """
    string = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", string)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", string).lower()


class QueryHelper:
    """Base class for query helpers."""

    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        """Initialize query helper."""
        self.provider = provider or config.ols_config.default_provider
        self.model = model or config.ols_config.default_model
