"""WatsonX provider implementation."""

import logging
from typing import Any

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import (
    GenTextParamsMetaNames as GenParams,
)
from langchain.llms.base import LLM
from langchain_ibm import WatsonxLLM

from ols import constants
from ols.src.llms.providers.provider import LLMProvider
from ols.src.llms.providers.registry import register_llm_provider_as

logger = logging.getLogger(__name__)


@register_llm_provider_as(constants.PROVIDER_WATSONX)
class WatsonX(LLMProvider):
    """WatsonX provider."""

    url: str = "https://us-south.ml.cloud.ibm.com"

    @property
    def default_params(self) -> dict[str, Any]:
        """Default LLM params."""
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

    def load(self) -> LLM:
        """Load LLM."""
        creds = {
            "url": self.provider_config.url or self.url,
            "apikey": self.provider_config.credentials,
        }

        llm_model = Model(
            model_id=self.model,
            credentials=creds,
            params=self.params,
            project_id=self.provider_config.project_id,
        )
        return WatsonxLLM(model=llm_model)
