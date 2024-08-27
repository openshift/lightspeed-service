"""Unit tests for RHOAI VLLM provider."""

import pytest
from langchain_openai.chat_models.base import ChatOpenAI

from ols.app.models.config import ProviderConfig
from ols.src.llms.providers.rhoai_vllm import RHOAIVLLM


@pytest.fixture
def provider_config():
    """Fixture with provider configuration for RHOAI VLLM."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "rhoai_vllm",
            "url": "test_url",
            "credentials_path": "tests/config/secret/apitoken",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test_model_url/",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )


@pytest.fixture
def provider_config_credentials_directory():
    """Fixture with provider configuration for RHOAI VLLM."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "rhoai_vllm",
            "url": "test_url",
            "credentials_path": "tests/config/secret",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test_model_url/",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )


@pytest.fixture
def provider_config_with_specific_parameters():
    """Fixture with provider configuration for RHOIAVLLM with specific parameters."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "rhoai_vllm",
            "url": "test_url",
            "credentials_path": "tests/config/secret/apitoken",
            "rhoai_vllm_config": {
                "url": "http://openai.com",
                "credentials_path": "tests/config/secret2/apitoken",
            },
            "models": [
                {
                    "name": "test_model_name",
                    "url": "http://test_model_url/",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )


def test_basic_interface(provider_config):
    """Test basic interface."""
    rhoai_vllm = RHOAIVLLM(model="uber-model", params={}, provider_config=provider_config)
    llm = rhoai_vllm.load()
    assert isinstance(llm, ChatOpenAI)
    assert rhoai_vllm.default_params
    assert "base_url" in rhoai_vllm.default_params
    assert "model" in rhoai_vllm.default_params
    assert "max_tokens" in rhoai_vllm.default_params


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
    }

    rhoai_vllm = RHOAIVLLM(model="uber-model", params=params, provider_config=provider_config)
    llm = rhoai_vllm.load()
    assert isinstance(llm, ChatOpenAI)
    assert rhoai_vllm.default_params
    assert rhoai_vllm.params

    # known parameters should be there
    assert "temperature" in rhoai_vllm.params
    assert "verbose" in rhoai_vllm.params
    assert rhoai_vllm.params["temperature"] == 0.3
    assert rhoai_vllm.params["verbose"] is True

    # unknown parameters should be filtered out
    assert "min_new_tokens" not in rhoai_vllm.params
    assert "max_new_tokens" not in rhoai_vllm.params
    assert "unknown_parameter" not in rhoai_vllm.params

    # taken from configuration
    assert rhoai_vllm.url == "test_url"
    assert rhoai_vllm.credentials == "secret_key"

    # API key should be loaded from secret
    assert rhoai_vllm.default_params["openai_api_key"] == "secret_key"

    assert rhoai_vllm.default_params["base_url"] == "test_url"


def test_credentials_key_in_directory_handling(provider_config_credentials_directory):
    """Test that credentials in directory is handled as expected."""
    params = {}

    rhoai_vllm = RHOAIVLLM(
        model="uber-model",
        params=params,
        provider_config=provider_config_credentials_directory,
    )
    llm = rhoai_vllm.load()
    assert isinstance(llm, ChatOpenAI)

    assert rhoai_vllm.credentials == "secret_key"


def test_loading_provider_specific_parameters(provider_config_with_specific_parameters):
    """Test that not allowed parameters are removed before model init."""
    rhoai_vllm = RHOAIVLLM(
        model="uber-model",
        params={},
        provider_config=provider_config_with_specific_parameters,
    )
    llm = rhoai_vllm.load()
    assert isinstance(llm, ChatOpenAI)
    assert rhoai_vllm.default_params
    assert rhoai_vllm.params

    assert "base_url" in rhoai_vllm.default_params
    assert "model" in rhoai_vllm.default_params
    assert "max_tokens" in rhoai_vllm.default_params

    # parameters taken from provier-specific configuration
    # which takes precedence over regular configuration
    assert rhoai_vllm.url == "http://openai.com/"
    assert rhoai_vllm.credentials == "secret_key_2"

    assert rhoai_vllm.default_params["openai_api_key"] == "secret_key_2"
    assert rhoai_vllm.default_params["base_url"] == "http://openai.com/"


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

    rhoai_vllm = RHOAIVLLM(model="uber-model", params=params, provider_config=provider_config)
    llm = rhoai_vllm.load()
    assert isinstance(llm, ChatOpenAI)
    assert rhoai_vllm.default_params
    assert rhoai_vllm.params

    # API key should be loaded from secret provided in specific param
    assert rhoai_vllm.default_params["openai_api_key"] == "secret_key"

    # base_url too should be read from specific params
    assert rhoai_vllm.default_params["base_url"] == "test_url"


def test_params_replace_default_values_with_none(provider_config):
    """Test if default values are replaced by None values."""
    # provider initialization with empty set of params
    rhoai_vllm = RHOAIVLLM(model="uber-model", params={}, provider_config=provider_config)
    rhoai_vllm.load()

    # check default value
    assert "base_url" in rhoai_vllm.params
    assert rhoai_vllm.params["base_url"] is not None

    # try to override default parameter
    params = {"base_url": None}

    rhoai_vllm = RHOAIVLLM(model="uber-model", params=params, provider_config=provider_config)
    rhoai_vllm.load()

    # known parameter(s) should be there, now with None values
    assert "base_url" in rhoai_vllm.params
    assert rhoai_vllm.params["base_url"] is None
