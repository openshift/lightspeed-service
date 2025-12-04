"""Base class for query helpers."""

import logging
from collections.abc import Callable
from typing import Optional

from langchain_core.language_models.llms import LLM

from ols import config
from ols.customize import prompts
from ols.src.llms.llm_loader import load_llm

logger = logging.getLogger(__name__)


class QueryHelper:
    """Base class for query helpers."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        generic_llm_params: Optional[dict] = None,
        llm_loader: Optional[Callable[[str, str, dict], LLM]] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        """Initialize query helper."""
        # NOTE: As signature of this method is evaluated before the config,
        # is loaded, we cannot use the config directly as defaults and we
        # need to use those values in the init evaluation.
        self.provider = provider or config.ols_config.default_provider
        self.model = model or config.ols_config.default_model
        self.generic_llm_params = generic_llm_params or {}
        self.llm_loader = llm_loader or load_llm

        self._system_prompt = (
            (config.dev_config.enable_system_prompt_override and system_prompt)
            or config.ols_config.system_prompt
            or prompts.QUERY_SYSTEM_INSTRUCTION
        )
        logger.debug("System prompt: %s", self._system_prompt)
