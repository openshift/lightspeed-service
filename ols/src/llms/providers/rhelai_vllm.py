"""Red Hat Enterprise Linux AI VLLM provider implementation."""

import logging
from typing import Any, Optional

from langchain.llms.base import LLM
from langchain_openai import ChatOpenAI

from ols import constants
from ols.src.llms.providers.provider import LLMProvider
from ols.src.llms.providers.registry import register_llm_provider_as

logger = logging.getLogger(__name__)


@register_llm_provider_as(constants.PROVIDER_RHELAI_VLLM)
class RHELAIVLLM(LLMProvider):
    """RHELAI VLLM provider."""

    # note: there's no default URL for RHELAI VLLM
    url: str = "https://api.openai.com/v1"
    credentials: Optional[str] = None

    @property
    def default_params(self) -> dict[str, Any]:
        """Construct and return structure with default LLM params."""
        self.url = str(self.provider_config.url or self.url)
        self.credentials = self.provider_config.credentials
        # provider-specific configuration has precendence over regular configuration
        if self.provider_config.rhelai_vllm_config is not None:
            rhelai_vllm_config = self.provider_config.rhelai_vllm_config
            self.url = str(rhelai_vllm_config.url)
            if rhelai_vllm_config.api_key is not None:
                self.credentials = rhelai_vllm_config.api_key

        return {
            "base_url": self.url,
            "openai_api_key": self.credentials,
            "model": self.model,
            "top_p": 0.95,
            "frequency_penalty": 1.03,
            "organization": None,
            "cache": None,
            "temperature": 0.01,
            "max_tokens": 512,
            "verbose": False,
            "http_client": self._construct_httpx_client(True),
        }

    def load(self) -> LLM:
        """Load LLM."""
        return ChatOpenAI(**self.params)
