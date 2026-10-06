"""OpenAI provider implementation."""

import logging
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ols import constants
from ols.app.models.config import ModelParameters
from ols.src.llms.llm_loader import LLMConfigurationError
from ols.src.llms.providers.provider import LLMProvider
from ols.src.llms.providers.registry import register_llm_provider_as
from ols.src.llms.providers.utils import populate_openai_reasoning

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
        self.credentials = self.provider_config.get_credentials()
        # provider-specific configuration has precendence over regular configuration
        if self.provider_config.openai_config is not None:
            openai_config = self.provider_config.openai_config
            self.url = str(openai_config.url)
            if openai_config.api_key is not None:
                self.credentials = openai_config.api_key

        model_config = self.provider_config.models.get(self.model)
        model_options = model_config.options or {} if model_config else {}
        use_responses_api = model_options.get("use_responses_api", True)
        if not isinstance(use_responses_api, bool):
            raise LLMConfigurationError("use_responses_api must be a boolean")

        default_parameters: dict[str, Any] = {
            "base_url": self.url,
            "openai_api_key": self.credentials,
            "model": self.model,
            "organization": None,
            "cache": None,
            "max_completion_tokens": 4096,
            "verbose": False,
            "use_responses_api": use_responses_api,
            "http_client": self._construct_httpx_client(False),
            "http_async_client": self._construct_httpx_client(True),
        }

        params = getattr(model_config, "parameters", None) or ModelParameters()
        populate_openai_reasoning(params, default_parameters)
        return default_parameters

    def load(self) -> BaseChatModel:
        """Load LLM with the configured API selection."""
        return ChatOpenAI(**self.params)
