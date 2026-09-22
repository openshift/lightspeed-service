"""Red Hat OpenShift VLLM provider implementation."""

import logging
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ols import constants
from ols.app.models.config import ModelParameters
from ols.src.llms.providers.chat_vlllm_reasoning import ChatVLLMReasoning
from ols.src.llms.providers.provider import LLMProvider
from ols.src.llms.providers.registry import register_llm_provider_as
from ols.src.llms.providers.utils import populate_vllm_reasoning

logger = logging.getLogger(__name__)


@register_llm_provider_as(constants.PROVIDER_RHOAI_VLLM)
class RHOAIVLLM(LLMProvider):
    """RHOAI VLLM provider."""

    # note: there's no default URL for RHOAI VLLM
    url: str = "https://api.openai.com/v1"
    credentials: Optional[str] = None

    @property
    def default_params(self) -> dict[str, Any]:
        """Construct and return structure with default LLM params."""
        self.url = str(self.provider_config.url or self.url)
        self.credentials = self.provider_config.get_credentials()
        # provider-specific configuration has precendence over regular configuration
        if self.provider_config.rhoai_vllm_config is not None:
            rhoai_vllm_config = self.provider_config.rhoai_vllm_config
            self.url = str(rhoai_vllm_config.url)
            if rhoai_vllm_config.api_key is not None:
                self.credentials = rhoai_vllm_config.api_key
        default_parameters: dict[str, Any] = {
            "base_url": self.url,
            "openai_api_key": self.credentials,
            "model": self.model,
            "top_p": 0.95,
            "frequency_penalty": 1.03,
            "organization": None,
            "cache": None,
            "temperature": 0.01,
            "max_completion_tokens": 4096,
            "verbose": False,
            "http_client": self._construct_httpx_client(False),
            "http_async_client": self._construct_httpx_client(True),
        }

        model_config = self.provider_config.models.get(self.model)
        params = getattr(model_config, "parameters", None) or ModelParameters()
        populate_vllm_reasoning(params, default_parameters)
        return default_parameters

    def load(self) -> BaseChatModel:
        """Load LLM."""
        model_config = self.provider_config.models.get(self.model)
        model_params = getattr(model_config, "parameters", None) or ModelParameters()
        reasoning_config = model_params.reasoning_config or {}

        # Use ChatVLLMReasoning when reasoning_config is present
        # to preserve reasoning_content/reasoning
        if reasoning_config:
            return ChatVLLMReasoning(**self.params)
        return ChatOpenAI(**self.params)
