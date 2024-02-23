"""LLM provider class definition."""

import abc
import logging
from typing import Any, Optional

from langchain.llms.base import LLM

from ols.app.models.config import ProviderConfig
from ols.utils import config

logger = logging.getLogger(__name__)


class AbstractLLMProvider(abc.ABC):
    """Abstract class defining `LLMProvider` interface."""

    @abc.abstractproperty
    def default_params(self) -> dict:
        """Defaults LLM params.

        These will be overriden by the input parameters of the caller or
        via developer config.
        """

    @abc.abstractmethod
    def load(self) -> LLM:
        """Load and langchain `LLM` instance and return it."""


class LLMProvider(AbstractLLMProvider):
    """LLM provider base class."""

    def __init__(
        self, model: str, provider_config: ProviderConfig, params: Optional[dict] = None
    ) -> None:
        """Initialize LLM provider.

        Args:
            model: The model name.
            provider_config: The provider configuration.
            params: The optional LLM parameters.
        """
        self.model = model
        self.provider_config = provider_config
        self.params = self._override_params(params or {})

    def _override_params(self, params: dict[Any, Any]) -> dict[Any, Any]:
        """Override LLM parameters if defined in developer config."""
        # input params overrides default params
        updated_params = {**self.default_params, **params}

        # config params overrides everything
        if config.dev_config.llm_params:
            logger.debug(
                f"overriding LLM params with debug options {config.dev_config.llm_params}"
            )
            updated_params = {**updated_params, **config.dev_config.llm_params}

        return updated_params
