"""BAM provider implementation."""

import logging

from genai import Client, Credentials
from genai.extensions.langchain import LangChainInterface
from genai.text.generation import TextGenerationParameters
from langchain.llms.base import LLM

from ols import constants
from ols.src.llms.providers.provider import LLMProvider
from ols.src.llms.providers.registry import register_llm_provider_as

logger = logging.getLogger(__name__)


@register_llm_provider_as(constants.PROVIDER_BAM)
class BAM(LLMProvider):
    """BAM provider."""

    @property
    def default_params(self):
        """Default LLM params."""
        params = {
            "decoding_method": "sample",
            "max_new_tokens": 512,
            "min_new_tokens": 1,
            "random_seed": 42,
            "top_k": 10,
            "top_p": 0.95,
            "repetition_penalty": 1.03,
            "temperature": 0.05,
        }
        return params

    def load(self) -> LLM:
        """Load LLM."""
        creds = Credentials(
            api_key=self.provider_config.credentials,
            api_endpoint=self.provider_config.url,
        )

        client = Client(credentials=creds)
        params = TextGenerationParameters(**self.params)

        llm = LangChainInterface(client=client, model_id=self.model, parameters=params)
        return llm
