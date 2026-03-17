"""Google Vertex AI provider implementation for Anthropic Claude models."""

import logging
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_vertexai.model_garden import ChatAnthropicVertex

from ols import constants
from ols.src.llms.providers.provider import LLMProvider
from ols.src.llms.providers.registry import register_llm_provider_as

logger = logging.getLogger(__name__)

DEFAULT_VERTEX_LOCATION = "us-east5"


@register_llm_provider_as(constants.PROVIDER_GOOGLE_VERTEX)
class GoogleVertex(LLMProvider):
    """Google Vertex AI provider for Anthropic Claude models."""

    project: Optional[str] = None
    location: str = DEFAULT_VERTEX_LOCATION

    @property
    def default_params(self) -> dict[str, Any]:
        """Construct and return structure with default LLM params."""
        self.project = self.provider_config.project_id
        if self.provider_config.google_vertex_config is not None:
            vertex_config = self.provider_config.google_vertex_config
            self.project = vertex_config.project
            self.location = vertex_config.location

        return {
            "model_name": self.model,
            "project": self.project,
            "location": self.location,
            "max_output_tokens": 512,
            "temperature": 0.01,
        }

    def load(self) -> BaseChatModel:
        """Load LLM."""
        return ChatAnthropicVertex(**self.params)
