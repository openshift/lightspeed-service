"""Azure OpenAI provider implementation."""

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from azure.core.credentials import AccessToken
from azure.identity import ClientSecretCredential
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI

from ols import constants
from ols.app.models.config import AzureOpenAIConfig
from ols.src.llms.providers.provider import LLMProvider
from ols.src.llms.providers.registry import register_llm_provider_as

logger = logging.getLogger(__name__)


TOKEN_EXPIRATION_LEEWAY = 30  # seconds


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


TOKEN_CACHE = TokenCache()  # per-process/worker cache


@register_llm_provider_as(constants.PROVIDER_AZURE_OPENAI)
class AzureOpenAI(LLMProvider):
    """Azure OpenAI provider."""

    url: str = "https://thiswillalwaysfail.openai.azure.com"
    credentials: Optional[str] = None

    @property
    def default_params(self) -> dict[str, Any]:
        """Construct and return structure with default LLM params."""
        self.url = str(self.provider_config.url or self.url)
        self.credentials = self.provider_config.credentials
        api_version = self.provider_config.api_version
        deployment_name = self.provider_config.deployment_name
        azure_config = self.provider_config.azure_config

        # provider-specific configuration has precendence over regular configuration
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
            "max_tokens": 512,
            "verbose": False,
            "http_client": self._construct_httpx_client(False, False),
            "http_async_client": self._construct_httpx_client(False, True),
        }

        # gpt-5 and o-series models don't support certain parameters
        if not ("gpt-5" in self.model or self.model.startswith("o")):
            default_parameters["temperature"] = 0.01
            default_parameters["top_p"] = 0.95
            default_parameters["frequency_penalty"] = 1.03

        if self.credentials is not None:
            # if credentials with API key is set, use it to call Azure OpenAI endpoints
            default_parameters["api_key"] = self.credentials
        else:
            # credentials for API key is not set -> azure AD token is
            # obtained through azure config parameters (tenant_id,
            # client_id and client_secret)
            access_token = self.resolve_access_token(azure_config)
            default_parameters["azure_ad_token"] = access_token
        return default_parameters

    def load(self) -> BaseChatModel:
        """Load LLM."""
        return AzureChatOpenAI(**self.params)

    def resolve_access_token(self, azure_config: AzureOpenAIConfig) -> Optional[str]:
        """Retrieve and cache Azure OpenAI access token."""
        if TOKEN_CACHE.is_expired():
            logger.info(
                "Cached AD token has expired (or missing) - generating a new one"
            )
            access_token = self.retrieve_access_token(azure_config)
            if access_token:
                TOKEN_CACHE.update_token(access_token.token, access_token.expires_on)
            else:
                return None
        return TOKEN_CACHE.access_token

    def retrieve_access_token(
        self, azure_config: AzureOpenAIConfig
    ) -> Optional[AccessToken]:
        """Retrieve access token to call Azure OpenAI."""
        if azure_config is None:
            raise ValueError(
                "Credentials for API token is not set and "
                "Azure-specific parameters are not provided."
                "It is not possible to retrieve access token."
            )
        if azure_config.tenant_id is None:
            raise_missing_attribute_error("tenant_id")
        if azure_config.client_id is None:
            raise_missing_attribute_error("client_id")
        if azure_config.client_secret is None:
            raise_missing_attribute_error("client_secret")

        # everything is there, try to retrieve credential
        try:
            credential = ClientSecretCredential(
                azure_config.tenant_id,
                azure_config.client_id,
                azure_config.client_secret,
            )
            return credential.get_token("https://cognitiveservices.azure.com/.default")
        except Exception as e:
            logger.error("Error retrieving access token: %s", e)
            return None


def raise_missing_attribute_error(attribute_name: str) -> None:
    """Raise exception when some attribute is missing in configuration."""
    raise ValueError(
        f"{attribute_name} should be set in azure_openai_config in order to retrieve access token."
    )
