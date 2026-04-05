"""Google Vertex AI provider for Anthropic Claude (Model Garden, not Gemini GenAI)."""

from typing import Any, Optional

from google.oauth2 import service_account
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_vertexai.model_garden import ChatAnthropicVertex

from ols import constants
from ols.src.llms.providers.provider import LLMProvider
from ols.src.llms.providers.registry import register_llm_provider_as
from ols.src.llms.providers.utils import VERTEX_AI_OAUTH_SCOPES, credentials_str_to_dict

DEFAULT_VERTEX_LOCATION = "us-east5"


@register_llm_provider_as(constants.PROVIDER_GOOGLE_VERTEX_ANTHROPIC)
class GoogleVertexAnthropic(LLMProvider):
    """Vertex AI provider for Anthropic Claude models (hosted by Anthropic)."""

    project: Optional[str] = None
    location: str = DEFAULT_VERTEX_LOCATION
    credentials: Optional[service_account.Credentials] = None

    @property
    def default_params(self) -> dict[str, Any]:
        """Construct and return structure with default LLM params."""
        self.project = self.provider_config.project_id
        if self.provider_config.credentials is None:
            raise ValueError(
                "credentials are required for Google Vertex Anthropic provider"
            )
        account_info = credentials_str_to_dict(self.provider_config.credentials)
        self.credentials = service_account.Credentials.from_service_account_info(
            account_info,
            scopes=VERTEX_AI_OAUTH_SCOPES,
        )  # type: ignore[no-untyped-call]
        if self.provider_config.google_vertex_anthropic_config is not None:
            vertex_config = self.provider_config.google_vertex_anthropic_config
            self.project = vertex_config.project
            self.location = vertex_config.location

        return {
            "model_name": self.model,
            "project": self.project,
            "location": self.location,
            "max_output_tokens": 512,
            "temperature": 0.01,
            "credentials": self.credentials,
        }

    def load(self) -> BaseChatModel:
        """Load LLM."""
        return ChatAnthropicVertex(**self.params)
