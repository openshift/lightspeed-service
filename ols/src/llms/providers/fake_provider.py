"""fake provider implementation."""

import logging
from typing import Any

from langchain.llms.base import LLM
from langchain_community.llms import FakeListLLM

from ols import constants
from ols.src.llms.providers.provider import LLMProvider
from ols.src.llms.providers.registry import register_llm_provider_as

logger = logging.getLogger(__name__)


@register_llm_provider_as(constants.PROVIDER_FAKE)
class FakeProvider(LLMProvider):
    """Fake provider for testing purposes."""

    @property
    def default_params(self) -> dict[str, Any]:
        """Construct and return structure with default LLM params."""
        return {}

    def load(self) -> LLM:
        """Load the fake LLM."""
        response = "This is a preconfigured fake response."
        return FakeListLLM(responses=[response])
