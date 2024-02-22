"""Unit tests for Azure OpenAI provider."""

import pytest
from langchain_openai import AzureChatOpenAI

from ols.app.models.config import ProviderConfig
from ols.src.llms.providers.azure_openai import AzureOpenAI
from ols.utils import config


@pytest.fixture
def provider_config():
    """Fixture with provider configuration for OpenAI."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "azure_openai",
            "url": "test_url",
            "credentials_path": "tests/config/secret.txt",
            "deployment_name": "test_deployment_name",
            "models": [
                {
                    "name": "test_model_name",
                }
            ],
        }
    )


def test_basic_interface(provider_config):
    """Test basic interface."""
    config.init_empty_config()  # needed for checking the config.dev_config.llm_params

    azure_openai = AzureOpenAI(
        model="uber-model", params={}, provider_config=provider_config
    )
    llm = azure_openai.load()
    assert isinstance(llm, AzureChatOpenAI)
    assert azure_openai.default_params
    assert "model" in azure_openai.default_params
    assert "deployment_name" in azure_openai.default_params
    assert "azure_endpoint" in azure_openai.default_params
    assert "max_tokens" in azure_openai.default_params


def test_params_handling(provider_config):
    """Test that not allowed parameters are removed before model init."""
    config.init_empty_config()  # needed for checking the config.dev_config.llm_params

    # these two parameters should be removed before model init
    params = {
        "min_new_tokens": 1,
        "max_new_tokens": 10,
    }

    azure_openai = AzureOpenAI(
        model="uber-model", params=params, provider_config=provider_config
    )
    llm = azure_openai.load()
    assert isinstance(llm, AzureChatOpenAI)
    assert azure_openai.default_params
    assert "min_new_tokens" not in azure_openai.default_params
    assert "max_new_tokens" not in azure_openai.default_params
