"""Question validators, statament classifiers, and response generators."""

from typing import Optional

from ols.utils import config


class QueryHelper:
    """Base class for query helpers."""

    def __init__(
        self, provider: Optional[str] = None, model: Optional[str] = None
    ) -> None:
        """Initialize query helper."""
        self.provider = provider or config.ols_config.default_provider
        self.model = model or config.ols_config.default_model
