"""Watsonx provider implementation."""

import logging
from typing import Any, Optional

from ibm_watsonx_ai.metanames import (
    GenTextParamsMetaNames as GenParams,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ibm import ChatWatsonx

from ols import constants
from ols.src.llms.providers.provider import LLMProvider
from ols.src.llms.providers.registry import register_llm_provider_as

logger = logging.getLogger(__name__)


@register_llm_provider_as(constants.PROVIDER_WATSONX)
class Watsonx(LLMProvider):
    """Watsonx provider."""

    url: str = "https://us-south.ml.cloud.ibm.com"
    credentials: Optional[str]
    project_id: Optional[str]

    @property
    def default_params(self) -> dict[str, Any]:
        """Construct and return structure with default LLM params."""
        # https://www.ibm.com/docs/en/watsonx-as-a-service?topic=models-parameters
        return {
            GenParams.DECODING_METHOD: "sample",
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.MAX_NEW_TOKENS: 512,
            GenParams.RANDOM_SEED: 42,
            GenParams.TEMPERATURE: 0.05,
            GenParams.TOP_K: 50,
            GenParams.TOP_P: 0.85,
            GenParams.REPETITION_PENALTY: 1.05,
        }

    def load(self) -> BaseChatModel:
        """Load LLM."""
        self.url = str(self.provider_config.url or self.url)
        self.credentials = self.provider_config.credentials
        self.project_id = self.provider_config.project_id

        # provider-specific configuration has precendence over regular configuration
        if self.provider_config.watsonx_config is not None:
            watsonx_config = self.provider_config.watsonx_config
            self.url = str(watsonx_config.url)
            self.project_id = watsonx_config.project_id
            if watsonx_config.api_key is not None:
                self.credentials = watsonx_config.api_key

        if self.credentials is None:
            raise ValueError("Credentials must be specified")

        if self.project_id is None:
            raise ValueError("Project ID must be specified")

        return ChatWatsonx(
            model_id=self.model,
            url=self.url,
            apikey=self.credentials,
            project_id=self.project_id,
            params=self.params,
        )
