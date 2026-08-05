"""Anthropic provider implementation."""

import logging
from typing import Any, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel

from ols import constants
from ols.src.llms.providers.provider import LLMProvider
from ols.src.llms.providers.registry import register_llm_provider_as

logger = logging.getLogger(__name__)


@register_llm_provider_as(constants.PROVIDER_ANTHROPIC)
class Anthropic(LLMProvider):
    """Anthropic provider."""

    url: str = "https://api.anthropic.com"
    credentials: Optional[str] = None

    @property
    def default_params(self) -> dict[str, Any]:
        """Construct and return structure with default LLM params."""
        self.url = str(self.provider_config.url or self.url)
        self.credentials = self.provider_config.credentials
        # provider-specific configuration has precedence over regular configuration
        if self.provider_config.anthropic_config is not None:
            anthropic_config = self.provider_config.anthropic_config
            self.url = str(anthropic_config.url)
            if anthropic_config.api_key is not None:
                self.credentials = anthropic_config.api_key

        default_parameters: dict[str, Any] = {
            "model": self.model,
            "anthropic_api_key": self.credentials,
            "base_url": self.url,
            "max_tokens": constants.DEFAULT_MAX_TOKENS_FOR_RESPONSE,
            "temperature": 0.01,
            "top_p": 0.95,
            "http_client": self._construct_httpx_client(True, False),
            "http_async_client": self._construct_httpx_client(True, True),
        }

        # Extended thinking support via model options reasoning_config
        model_config = self.provider_config.models.get(self.model)
        if model_config and model_config.options:
            reasoning_config = model_config.options.get("reasoning_config")
            if reasoning_config:
                default_parameters["thinking"] = reasoning_config
                # Anthropic requires default temperature when thinking is enabled
                default_parameters.pop("temperature", None)
                default_parameters.pop("top_p", None)

        return default_parameters

    def load(self) -> BaseChatModel:
        """Load LLM."""
        return ChatAnthropic(**self.params)
