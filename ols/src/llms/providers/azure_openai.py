"""Azure OpenAI provider implementation."""

import logging
from typing import Any, Optional

from langchain.llms.base import LLM
from langchain_openai import AzureChatOpenAI

from ols import constants
from ols.src.llms.providers.provider import LLMProvider
from ols.src.llms.providers.registry import register_llm_provider_as

logger = logging.getLogger(__name__)


@register_llm_provider_as(constants.PROVIDER_AZURE_OPENAI)
class AzureOpenAI(LLMProvider):
    """Azure OpenAI provider."""

    url: str = "https://thiswillalwaysfail.openai.azure.com"
    credentials: Optional[str] = None

    @property
    def default_params(self) -> dict[str, Any]:
        """Default LLM params."""
        self.url = self.provider_config.url or self.url
        self.credentials = self.provider_config.credentials
        deployment_name = self.provider_config.deployment_name

        # provider-specific configuration has precendence over regular configuration
        if self.provider_config.azure_config is not None:
            azure_config = self.provider_config.azure_config
            self.url = str(azure_config.url)
            deployment_name = azure_config.deployment_name
            self.credentials = azure_config.api_key

        return {
            "azure_endpoint": self.url,
            "api_key": self.credentials,
            "api_version": "2024-02-01",
            "deployment_name": deployment_name,
            "model": self.model,
            "model_kwargs": {
                "top_p": 0.95,
                "frequency_penalty": 1.03,
            },
            "organization": None,
            "cache": None,
            "streaming": True,
            "temperature": 0.01,
            "max_tokens": 512,
            "verbose": False,
        }

    def load(self) -> LLM:
        """Load LLM."""
        return AzureChatOpenAI(**self.params)
