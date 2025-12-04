"""fake provider implementation."""

import logging
from typing import Any

from langchain_community.llms import FakeListLLM
from langchain_community.llms.fake import FakeStreamingListLLM
from langchain_core.language_models.llms import LLM

from ols import constants
from ols.src.llms.providers.provider import LLMProvider
from ols.src.llms.providers.registry import register_llm_provider_as

logger = logging.getLogger(__name__)


@register_llm_provider_as(constants.PROVIDER_FAKE)
class FakeProvider(LLMProvider):
    """Fake provider for testing purposes."""

    stream: bool = False
    response: str = "This is a preconfigured fake response."
    chunks: int = len(response)
    sleep: float = 0.1

    @property
    def default_params(self) -> dict[str, Any]:
        """Construct and return structure with default LLM params."""
        if self.provider_config.fake_provider_config is not None:
            fake_provider_config = self.provider_config.fake_provider_config
            if fake_provider_config.stream:
                self.stream = fake_provider_config.stream
            if fake_provider_config.response:
                self.response = fake_provider_config.response
            if fake_provider_config.chunks:
                self.chunks = fake_provider_config.chunks
            if fake_provider_config.sleep:
                self.sleep = fake_provider_config.sleep

        return {
            "stream": self.stream,
            "response": self.response,
            "chunks": self.chunks,
            "sleep": self.sleep,
        }

    def load(self) -> LLM:
        """Load the fake LLM."""
        if self.stream:
            i = self.chunks // (len(self.response) + 1)
            j = self.chunks % (len(self.response) + 1)
            response = ((self.response + " ") * i) + self.response[0:j]
            return FakeStreamingListLLM(responses=[response], sleep=self.sleep)

        response = self.response
        return FakeListLLM(responses=[response])
