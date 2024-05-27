"""Unit tests for Azure OpenAI provider."""

import pytest
from langchain_openai import AzureChatOpenAI

from ols.app.models.config import ProviderConfig
from ols.src.llms.providers.azure_openai import AzureOpenAI


@pytest.fixture
def provider_config():
    """Fixture with provider configuration for Azure OpenAI."""
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


@pytest.fixture
def provider_config_with_specific_parameters():
    """Fixture with provider configuration for Azure OpenAI with specific parameters."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "azure_openai",
            "url": "test_url",
            "credentials_path": "tests/config/secret.txt",
            "deployment_name": "test_deployment_name",
            "azure_openai_config": {
                "url": "http://azure.com",
                "deployment_name": "azure_deployment_name",
                "credentials_path": "tests/config/secret2.txt",
                "tenant_id": "00000000-0000-0000-0000-000000000001",
                "client_id": "00000000-0000-0000-0000-000000000002",
                "client_secret_path": "tests/config/secret.txt",
            },
            "models": [
                {
                    "name": "test_model_name",
                }
            ],
        }
    )


def test_basic_interface(provider_config):
    """Test basic interface."""
    azure_openai = AzureOpenAI(
        model="uber-model", params={}, provider_config=provider_config
    )
    llm = azure_openai.load()
    assert isinstance(llm, AzureChatOpenAI)
    assert azure_openai.default_params

    # parameter presence test
    assert "model" in azure_openai.default_params
    assert "deployment_name" in azure_openai.default_params
    assert "api_key" in azure_openai.default_params
    assert "azure_endpoint" in azure_openai.default_params
    assert "max_tokens" in azure_openai.default_params
    assert "api_version" in azure_openai.default_params

    # test parameter values taken from config
    assert azure_openai.default_params["deployment_name"] == "test_deployment_name"

    # API key should be loaded from secret
    assert azure_openai.default_params["api_key"] == "secret_key"

    assert azure_openai.default_params["azure_endpoint"] == "test_url"


def test_loading_provider_specific_parameters(provider_config_with_specific_parameters):
    """Test if provider-specific parameters are loaded too."""
    azure_openai = AzureOpenAI(
        model="uber-model",
        params={},
        provider_config=provider_config_with_specific_parameters,
    )
    llm = azure_openai.load()
    assert isinstance(llm, AzureChatOpenAI)
    assert azure_openai.default_params

    # parameter presence test
    assert "model" in azure_openai.default_params
    assert "deployment_name" in azure_openai.default_params
    assert "api_key" in azure_openai.default_params
    assert "azure_endpoint" in azure_openai.default_params
    assert "max_tokens" in azure_openai.default_params
    assert "api_version" in azure_openai.default_params

    # test parameter values taken from provider-specific config
    assert azure_openai.default_params["deployment_name"] == "azure_deployment_name"
    assert azure_openai.default_params["azure_endpoint"] == "http://azure.com/"

    # API key should be loaded from secret
    assert azure_openai.default_params["api_key"] == "secret_key_2"


def test_params_handling(provider_config):
    """Test that not allowed parameters are removed before model init."""
    # first three parameters should be removed before model init
    # rest need to stay
    params = {
        "unknown_parameter": "foo",
        "min_new_tokens": 1,
        "max_new_tokens": 10,
        "temperature": 0.3,
        "verbose": True,
        "api_version": "2023-12-31",
    }

    azure_openai = AzureOpenAI(
        model="uber-model", params=params, provider_config=provider_config
    )
    llm = azure_openai.load()
    assert isinstance(llm, AzureChatOpenAI)
    assert azure_openai.default_params
    assert azure_openai.params

    # known parameters should be there
    assert "temperature" in azure_openai.params
    assert "verbose" in azure_openai.params
    assert "api_version" in azure_openai.params
    assert azure_openai.params["temperature"] == 0.3
    assert azure_openai.params["verbose"] is True
    assert azure_openai.params["api_version"] == "2023-12-31"

    # unknown parameters should be filtered out
    assert "min_new_tokens" not in azure_openai.params
    assert "max_new_tokens" not in azure_openai.params
    assert "unknown_parameter" not in azure_openai.params


def test_api_version_can_not_be_none(provider_config):
    """Test that api_version parameter can not be None."""
    params = {
        "api_version": None,
    }

    azure_openai = AzureOpenAI(
        model="uber-model", params=params, provider_config=provider_config
    )

    # api_version is required parameter and can not be None
    with pytest.raises(KeyError, match="api_version"):
        azure_openai.load()


def test_none_params_handling(provider_config):
    """Test that not allowed parameters are removed before model init."""
    # first three parameters should be removed before model init
    # rest need to stay
    params = {
        "unknown_parameter": None,
        "min_new_tokens": None,
        "max_new_tokens": None,
        "organization": None,
        "cache": None,
    }

    azure_openai = AzureOpenAI(
        model="uber-model", params=params, provider_config=provider_config
    )
    llm = azure_openai.load()
    assert isinstance(llm, AzureChatOpenAI)
    assert azure_openai.default_params
    assert azure_openai.params

    # known parameters should be there
    assert "organization" in azure_openai.params
    assert "cache" in azure_openai.params
    assert azure_openai.params["organization"] is None
    assert azure_openai.params["cache"] is None

    # unknown parameters should be filtered out
    assert "min_new_tokens" not in azure_openai.params
    assert "max_new_tokens" not in azure_openai.params
    assert "unknown_parameter" not in azure_openai.params
