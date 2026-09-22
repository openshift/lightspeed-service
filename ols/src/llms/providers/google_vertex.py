"""Google Vertex AI providers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai.model_garden import ChatAnthropicVertex

from ols import constants
from ols.app.models.config import ModelParameters
from ols.src.llms.providers.provider import LLMProvider
from ols.src.llms.providers.registry import register_llm_provider_as
from ols.src.llms.providers.utils import load_vertex_credentials
from ols.utils.checks import InvalidConfigurationError

if TYPE_CHECKING:
    from google.auth.credentials import Credentials as GoogleCredentials
    from langchain_core.language_models.chat_models import BaseChatModel

DEFAULT_VERTEX_GEMINI_LOCATION = "global"
DEFAULT_VERTEX_ANTHROPIC_LOCATION = "us-east5"


@register_llm_provider_as(constants.PROVIDER_GOOGLE_VERTEX)
class GoogleVertex(LLMProvider):
    """Vertex AI provider for Gemini and other models hosted by Google."""

    project: Optional[str] = None
    location: str = DEFAULT_VERTEX_GEMINI_LOCATION
    credentials: Optional[GoogleCredentials] = None

    @property
    def default_params(self) -> dict[str, Any]:
        """Construct and return structure with default LLM params."""
        self.project = self.provider_config.project_id
        creds_value = self.provider_config.get_credentials()
        if creds_value is None:
            raise InvalidConfigurationError(
                "credentials are required for Google Vertex provider"
            )
        self.credentials = load_vertex_credentials(creds_value)
        if self.provider_config.google_vertex_config is not None:
            vertex_config = self.provider_config.google_vertex_config
            self.project = vertex_config.project
            self.location = vertex_config.location

        default_params: dict[str, Any] = {
            "model": self.model,
            "project": self.project,
            "location": self.location,
            "vertexai": True,
            "max_output_tokens": 512,
            "temperature": 0.01,
            "credentials": self.credentials,
        }
        if self.provider_config.url is not None:
            default_params["base_url"] = str(self.provider_config.url)

        model_config = self.provider_config.models.get(self.model)
        model_params = getattr(model_config, "parameters", None) or ModelParameters()
        reasoning_config = model_params.reasoning_config or {}
        if reasoning_config:
            if "include_thoughts" in reasoning_config:
                default_params["include_thoughts"] = reasoning_config[
                    "include_thoughts"
                ]
            if "thinking_level" in reasoning_config:
                default_params["thinking_level"] = reasoning_config["thinking_level"]
            if "thinking_budget" in reasoning_config:
                default_params["thinking_budget"] = reasoning_config["thinking_budget"]

        return default_params

    def load(self) -> BaseChatModel:
        """Load LLM."""
        return ChatGoogleGenerativeAI(**self.params)


@register_llm_provider_as(constants.PROVIDER_GOOGLE_VERTEX_ANTHROPIC)
class GoogleVertexAnthropic(LLMProvider):
    """Vertex AI provider for Anthropic Claude models (hosted by Anthropic)."""

    project: Optional[str] = None
    location: str = DEFAULT_VERTEX_ANTHROPIC_LOCATION
    credentials: Optional[GoogleCredentials] = None

    @property
    def default_params(self) -> dict[str, Any]:
        """Construct and return structure with default LLM params."""
        self.project = self.provider_config.project_id
        creds_value = self.provider_config.get_credentials()
        if creds_value is None:
            raise InvalidConfigurationError(
                "credentials are required for Google Vertex Anthropic provider"
            )
        self.credentials = load_vertex_credentials(creds_value)
        if self.provider_config.google_vertex_anthropic_config is not None:
            vertex_config = self.provider_config.google_vertex_anthropic_config
            self.project = vertex_config.project
            self.location = vertex_config.location

        default_params: dict[str, Any] = {
            "model_name": self.model,
            "project": self.project,
            "location": self.location,
            "max_output_tokens": 512,
            "temperature": 0.01,
            "credentials": self.credentials,
        }

        model_config = self.provider_config.models.get(self.model)
        model_params = getattr(model_config, "parameters", None) or ModelParameters()
        reasoning_config = model_params.reasoning_config or {}
        if reasoning_config and reasoning_config.get("thinking_enabled"):
            thinking_config = {}
            if reasoning_config.get("thinking_effort") in ["high", "xhigh", "max"]:
                thinking_config["type"] = "adaptive"
                thinking_config["display"] = "summarized"
            else:
                thinking_config["type"] = "enabled"
                budget = reasoning_config.get("budget_tokens", 0)
                if budget > 0:
                    thinking_config["budget_tokens"] = budget
            default_params["model_kwargs"] = {"thinking": thinking_config}

        return default_params

    def load(self) -> BaseChatModel:
        """Load LLM."""
        return ChatAnthropicVertex(**self.params)
