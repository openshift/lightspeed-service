"""Base class for query helpers."""

import logging
from typing import Callable, Optional

from langchain.llms.base import LLM

from ols.src.llms.llm_loader import load_llm
from ols.utils import config

logger = logging.getLogger(__name__)


class QueryHelper:
    """Base class for query helpers."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        llm_params: Optional[dict] = None,
        llm_loader: Optional[Callable[[str, str, dict], LLM]] = None,
    ) -> None:
        """Initialize query helper."""
        # NOTE: As signature of this method is evaluated before the config,
        # is loaded, we cannot use the config directly as defaults and we
        # need to use those values in the init evaluation.
        self.provider = provider or config.ols_config.default_provider
        self.model = model or config.ols_config.default_model
        self.llm_params = llm_params or {}
        self.llm_loader = llm_loader or load_llm
