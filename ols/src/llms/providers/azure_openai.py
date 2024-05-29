"""Azure OpenAI provider implementation."""

import logging
from typing import Any, Optional

from azure.core.credentials import AccessToken
from azure.identity import ClientSecretCredential
from langchain.llms.base import LLM
from langchain_openai import AzureChatOpenAI

from ols import constants
from ols.app.models.config import AzureOpenAIConfig
from ols.src.llms.providers.provider import LLMProvider
from ols.src.llms.providers.registry import register_llm_provider_as

logger = logging.getLogger(__name__)


@register_llm_provider_as(constants.PROVIDER_AZURE_OPENAI)
class AzureOpenAI(LLMProvider):
    """Azure OpenAI provider."""

    url: str = "https://thiswillalwaysfail.openai.azure.com"
    credentials: Optional[str] = None

    @property
    def default_params(self) -> dict[str, Any]:
        """Default LLM params."""
        self.url = str(self.provider_config.url or self.url)
        self.credentials = self.provider_config.credentials
        deployment_name = self.provider_config.deployment_name
        azure_config = self.provider_config.azure_config

        # provider-specific configuration has precendence over regular configuration
        if azure_config is not None:
            self.url = str(azure_config.url)
            deployment_name = azure_config.deployment_name
            self.credentials = azure_config.api_key

        default_parameters = {
            "azure_endpoint": self.url,
            "api_version": "2024-02-01",
            "deployment_name": deployment_name,
            "model": self.model,
            "model_kwargs": {
                "top_p": 0.95,
                "frequency_penalty": 1.03,
            },
            "organization": None,
            "cache": None,
            "streaming": True,
            "temperature": 0.01,
            "max_tokens": 512,
            "verbose": False,
        }

        if self.credentials is not None:
            # if credentials with API key is set, use it to call Azure OpenAI endpoints
            default_parameters["api_key"] = self.credentials
        else:
            # credentials for API key is not set
            # -> it is needed to retrieve token first
            access_token = self.retrieve_access_token(azure_config)
            # set up active directory token to access Azure services, including OpenAI one
            default_parameters["azure_ad_token"] = access_token.token

        return default_parameters

    def load(self) -> LLM:
        """Load LLM."""
        return AzureChatOpenAI(**self.params)

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
            logger.error(f"Error retrieving access token: {e}")
            return None


def raise_missing_attribute_error(attribute_name: str) -> None:
    """Raise exception when some attribute is missing in configuration."""
    raise ValueError(
        f"{attribute_name} should be set in azure_openai_config in order to retrieve "
        "access token."
    )
