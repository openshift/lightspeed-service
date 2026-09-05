"""OpenAI provider implementation."""

import logging
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ols import constants
from ols.app.models.config import ModelParameters
from ols.src.llms.providers.provider import LLMProvider
from ols.src.llms.providers.registry import register_llm_provider_as

logger = logging.getLogger(__name__)


@register_llm_provider_as(constants.PROVIDER_OPENAI)
class OpenAI(LLMProvider):
    """OpenAI provider."""

    url: str = "https://api.openai.com/v1"
    credentials: Optional[str] = None

    @property
    def default_params(self) -> dict[str, Any]:
        """Construct and return structure with default LLM params."""
        self.url = str(self.provider_config.url or self.url)
        self.credentials = self.provider_config.credentials
        # provider-specific configuration has precendence over regular configuration
        if self.provider_config.openai_config is not None:
            openai_config = self.provider_config.openai_config
            self.url = str(openai_config.url)
            if openai_config.api_key is not None:
                self.credentials = openai_config.api_key

        default_parameters: dict[str, Any] = {
            "base_url": self.url,
            "openai_api_key": self.credentials,
            "model": self.model,
            "organization": None,
            "cache": None,
            "max_completion_tokens": 512,
            "verbose": False,
            "http_client": self._construct_httpx_client(False),
            "http_async_client": self._construct_httpx_client(True),
        }

        model_config = self.provider_config.models.get(self.model)
        params = getattr(model_config, "parameters", None) or ModelParameters()
        reasoning_config = params.reasoning_config or {}

        if reasoning_config:
            reasoning_parameters = dict(reasoning_config)
            if "verbosity" in reasoning_parameters:
                default_parameters["verbosity"] = reasoning_parameters.pop("verbosity")
            default_parameters["reasoning"] = reasoning_parameters
        return default_parameters

    def load(self) -> BaseChatModel:
        """Load LLM."""
        return ChatOpenAI(**self.params)
