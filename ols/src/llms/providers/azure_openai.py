"""Azure OpenAI provider implementation."""

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from azure.core.credentials import AccessToken
from azure.identity import ClientSecretCredential
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from ols import config, constants
from ols.app.models.config import AzureOpenAIConfig, ModelParameters, ProviderConfig
from ols.src.llms.llm_loader import LLMConfigurationError
from ols.src.llms.providers.provider import LLMProvider
from ols.src.llms.providers.registry import register_llm_provider_as
from ols.src.llms.providers.utils import populate_openai_reasoning

logger = logging.getLogger(__name__)


TOKEN_EXPIRATION_LEEWAY = 30  # seconds
COGNITIVE_SERVICES_TOKEN_SCOPE = (
    "https://cognitiveservices.azure.com/.default"  # noqa: S105
)
AZURE_AI_TOKEN_SCOPE = "https://ai.azure.com/.default"  # noqa: S105


@dataclass
class TokenCache:
    """Token cache for Azure OpenAI provider."""

    access_token: Optional[str] = None
    expires_on: int = 0

    def is_expired(self) -> bool:
        """Check if token has expired."""
        return self.expires_on == 0 or time.time() > self.expires_on

    def update_token(self, token: str, expires_on: int) -> None:
        """Update token and expiration time."""
        self.access_token = token
        self.expires_on = expires_on - TOKEN_EXPIRATION_LEEWAY


TOKEN_CACHES: dict[tuple[str, str, str, str], TokenCache] = {}


@register_llm_provider_as(constants.PROVIDER_AZURE_OPENAI)
class AzureOpenAI(LLMProvider):
    """Azure OpenAI provider."""

    url: str = "https://thiswillalwaysfail.openai.azure.com"
    credentials: Optional[str] = None

    def __init__(
        self,
        model: str,
        provider_config: ProviderConfig,
        params: Optional[dict] = None,
    ) -> None:
        """Select the authentication scope before resolving default parameters."""
        model_config = provider_config.models.get(model)
        model_options = model_config.options or {} if model_config else {}
        self._use_responses_api = config.dev_config.llm_params.get(
            "use_responses_api",
            (params or {}).get(
                "use_responses_api", model_options.get("use_responses_api", True)
            ),
        )
        if not isinstance(self._use_responses_api, bool):
            raise LLMConfigurationError("use_responses_api must be a boolean")
        super().__init__(model, provider_config, params)

    @property
    def default_params(self) -> dict[str, Any]:
        """Construct and return structure with default LLM params."""
        self.url = str(self.provider_config.url or self.url)
        self.credentials = self.provider_config.get_credentials()
        api_version = self.provider_config.api_version
        deployment_name = self.provider_config.deployment_name
        azure_config = self.provider_config.azure_config

        # provider-specific configuration has precedence over regular configuration
        if azure_config is not None:
            self.url = str(azure_config.url)
            deployment_name = azure_config.deployment_name
            if azure_config.api_key is not None:
                self.credentials = azure_config.api_key

        default_parameters: dict[str, Any] = {
            "azure_endpoint": self.url,
            "api_version": api_version,
            "deployment_name": deployment_name,
            "model": self.model,
            "organization": None,
            "cache": None,
            "max_completion_tokens": 4096,
            "verbose": False,
            "use_responses_api": self._use_responses_api,
            "http_client": self._construct_httpx_client(False),
            "http_async_client": self._construct_httpx_client(True),
        }

        model_config = self.provider_config.models.get(self.model)
        params = getattr(model_config, "parameters", None) or ModelParameters()
        populate_openai_reasoning(params, default_parameters)

        if self.credentials is not None:
            # if credentials with API key is set, use it to call Azure OpenAI endpoints
            default_parameters["api_key"] = self.credentials
        else:
            # credentials for API key is not set -> azure AD token is
            # obtained through azure config parameters (tenant_id,
            # client_id and client_secret)
            access_token = self.resolve_access_token(azure_config)
            default_parameters["azure_ad_token"] = access_token
        params_to_redact = {
            "api_key",
            "azure_ad_token",
            "http_client",
            "http_async_client",
        }
        logger.info(
            "Created Azure default parameters %s",
            {
                k: "***" if k in params_to_redact else v
                for k, v in default_parameters.items()
            },
        )
        return default_parameters

    def load(self) -> BaseChatModel:
        """Load LLM using Responses API unless Chat Completions is requested."""
        params = dict(self.params)
        if params.get("use_responses_api"):
            azure_endpoint = params.pop("azure_endpoint").rstrip("/")
            params.pop("api_version", None)
            deployment_name = params.pop("deployment_name")
            api_key = params.pop("api_key", None)
            azure_ad_token = params.pop("azure_ad_token", None)
            params["base_url"] = f"{azure_endpoint}/openai/v1/"
            params["model"] = deployment_name
            params["openai_api_key"] = (
                api_key if api_key is not None else azure_ad_token
            )
            return ChatOpenAI(**params)
        return AzureChatOpenAI(**params)

    def resolve_access_token(self, azure_config: AzureOpenAIConfig) -> str:
        """Retrieve and cache Azure OpenAI access token."""
        if azure_config is None:
            raise LLMConfigurationError(
                "Credentials for API token is not set and "
                "Azure-specific parameters are not provided."
            )
        scope = (
            AZURE_AI_TOKEN_SCOPE
            if self._use_responses_api
            else COGNITIVE_SERVICES_TOKEN_SCOPE
        )
        secret_hash = hashlib.sha256(
            (azure_config.client_secret or "").encode()
        ).hexdigest()
        cache_key = (
            azure_config.tenant_id or "",
            azure_config.client_id or "",
            secret_hash,
            scope,
        )
        cache = TOKEN_CACHES.setdefault(cache_key, TokenCache())
        if cache.is_expired():
            logger.info(
                "Cached AD token has expired (or missing) - generating a new one"
            )
            access_token = self.retrieve_access_token(azure_config)
            cache.update_token(access_token.token, access_token.expires_on)
        return cache.access_token

    def retrieve_access_token(self, azure_config: AzureOpenAIConfig) -> AccessToken:
        """Retrieve access token to call Azure OpenAI."""
        if azure_config is None:
            raise LLMConfigurationError(
                "Credentials for API token is not set and "
                "Azure-specific parameters are not provided. "
                "It is not possible to retrieve access token."
            )
        if azure_config.tenant_id is None:
            raise_missing_attribute_error("tenant_id")
        if azure_config.client_id is None:
            raise_missing_attribute_error("client_id")
        if azure_config.client_secret is None:
            raise_missing_attribute_error("client_secret")

        try:
            credential = ClientSecretCredential(
                azure_config.tenant_id,
                azure_config.client_id,
                azure_config.client_secret,
            )
            scope = (
                AZURE_AI_TOKEN_SCOPE
                if self._use_responses_api
                else COGNITIVE_SERVICES_TOKEN_SCOPE
            )
            return credential.get_token(scope)
        except Exception as e:
            logger.error("Failed to acquire Azure Entra ID access token: %s", e)
            raise LLMConfigurationError(
                f"Failed to acquire Azure Entra ID access token: {e}"
            ) from e


def raise_missing_attribute_error(attribute_name: str) -> None:
    """Raise exception when some attribute is missing in configuration."""
    raise LLMConfigurationError(
        f"{attribute_name} should be set in azure_openai_config in order to retrieve access token."
    )
