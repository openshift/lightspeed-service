"""BAM provider implementation."""

import logging
from typing import Any, Optional

from genai import Client, Credentials
from genai.extensions.langchain import LangChainInterface
from genai.schema import TextGenerationParameters
from langchain_core.language_models.llms import LLM

from ols import constants
from ols.src.llms.providers.provider import LLMProvider
from ols.src.llms.providers.registry import register_llm_provider_as

logger = logging.getLogger(__name__)


@register_llm_provider_as(constants.PROVIDER_BAM)
class BAM(LLMProvider):
    """BAM provider."""

    url: str = "https://bam-api.res.ibm.com"
    credentials: Optional[str]

    @property
    def default_params(self) -> dict[str, Any]:
        """Construct and return structure with default LLM params."""
        return {
            "decoding_method": "sample",
            "max_new_tokens": 512,
            "min_new_tokens": 1,
            "random_seed": 42,
            "top_k": 50,
            "top_p": 0.85,
            "repetition_penalty": 1.05,
            "temperature": 0.05,
        }

    def load(self) -> LLM:
        """Load LLM."""
        self.url = str(self.provider_config.url or self.url)
        self.credentials = self.provider_config.credentials

        # provider-specific configuration has precendence over regular configuration
        if self.provider_config.bam_config is not None:
            bam_config = self.provider_config.bam_config
            self.url = str(bam_config.url)
            if bam_config.api_key is not None:
                self.credentials = bam_config.api_key

        if self.credentials is None:
            raise ValueError("Credentials must be specified")

        creds = Credentials(
            api_key=self.credentials,
            api_endpoint=self.url,
        )

        client = Client(credentials=creds)
        params = TextGenerationParameters(**self.params)

        return LangChainInterface(client=client, model_id=self.model, parameters=params)
