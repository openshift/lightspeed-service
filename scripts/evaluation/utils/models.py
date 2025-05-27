"""Models for evaluation."""

from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

from ols.src.llms.providers.azure_openai import AzureOpenAI
from ols.src.llms.providers.openai import OpenAI
from ols.src.llms.providers.watsonx import Watsonx


class OpenAIVanilla(OpenAI):
    """OpenAI provider."""

    # pylint: disable=W0201
    @property
    def default_params(self):
        """Construct and return structure with default LLM params."""
        self.url = str(self.provider_config.url)
        self.credentials = self.provider_config.credentials
        # provider-specific configuration has precendence over regular configuration
        if self.provider_config.openai_config is not None:
            openai_config = self.provider_config.openai_config
            self.url = str(openai_config.url)
            if openai_config.api_key is not None:
                self.credentials = openai_config.api_key

        return {
            "base_url": self.url,
            "openai_api_key": self.credentials,
            "model": self.model,
        }


class AzureOpenAIVanilla(AzureOpenAI):
    """Azure OpenAI provider."""

    # pylint: disable=W0201
    @property
    def default_params(self):
        """Construct and return structure with default LLM params."""
        self.url = str(self.provider_config.url or self.url)
        self.credentials = self.provider_config.credentials
        deployment_name = self.provider_config.deployment_name
        azure_config = self.provider_config.azure_config

        # provider-specific configuration has precendence over regular configuration
        if azure_config is not None:
            self.url = str(azure_config.url)
            deployment_name = azure_config.deployment_name
            if azure_config.api_key is not None:
                self.credentials = azure_config.api_key

        default_parameters = {
            "azure_endpoint": self.url,
            "api_version": "2024-02-15-preview",
            "deployment_name": deployment_name,
            "model": self.model,
        }

        if self.credentials is not None:
            # if credentials with API key is set, use it to call Azure OpenAI endpoints
            default_parameters["api_key"] = self.credentials
        else:
            # credentials for API key is not set -> azure AD token is
            # obtained through azure config parameters (tenant_id,
            # client_id and client_secret)
            assert azure_config is not None, "Azure OpenAI configuration is missing"
            access_token = self.resolve_access_token(azure_config)
            default_parameters["azure_ad_token"] = access_token
        return default_parameters


class WatsonxVanilla(Watsonx):
    """Watsonx provider."""

    @property
    def default_params(self):
        """Construct and return structure with default LLM params."""
        return {
            GenTextParamsMetaNames.MAX_NEW_TOKENS: 4096,
        }


VANILLA_MODEL = {
    "watsonx": WatsonxVanilla,
    "openai": OpenAIVanilla,
    "azure_openai": AzureOpenAIVanilla,
}

MODEL_OLS_PARAM = {"watsonx": Watsonx, "openai": OpenAI, "azure_openai": AzureOpenAI}
