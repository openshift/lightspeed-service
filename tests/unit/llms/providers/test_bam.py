"""Unit tests for BAM provider."""

import pytest
from genai.extensions.langchain import LangChainInterface

from ols.app.models.config import ProviderConfig
from ols.src.llms.providers.bam import BAM


@pytest.fixture
def provider_config():
    """Fixture with provider configuration for BAM."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "bam",
            "url": "test_url",
            "credentials_path": "tests/config/secret/apitoken",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "test_model_url",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )


@pytest.fixture
def provider_config_credentials_directory():
    """Fixture with provider configuration for BAM."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "bam",
            "url": "test_url",
            "credentials_path": "tests/config/secret",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "test_model_url",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )


@pytest.fixture
def provider_config_without_credentials():
    """Fixture with provider configuration for BAM without credentials."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "bam",
            "url": "test_url",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "test_model_url",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )


@pytest.fixture
def provider_config_with_specific_params():
    """Fixture with provider configuration for BAM."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "bam",
            "url": "test_url",
            "credentials_path": "tests/config/secret/apitoken",
            "bam_config": {
                "url": "http://bam.com",
                "credentials_path": "tests/config/secret2/apitoken",
            },
            "models": [
                {
                    "name": "test_model_name",
                    "url": "test_model_url",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )


def test_basic_interface(provider_config):
    """Test basic interface."""
    bam = BAM(model="uber-model", params={}, provider_config=provider_config)
    llm = bam.load()
    assert isinstance(llm, LangChainInterface)
    assert bam.default_params


def test_params_handling(provider_config):
    """Test that not allowed parameters are removed before model init."""
    # first two parameters should be removed before model init
    # rest need to stay
    params = {
        "unknown_parameter": "foo",
        "verbose": True,
        "min_new_tokens": 1,
        "max_new_tokens": 10,
        "temperature": 0.3,
    }

    bam = BAM(model="uber-model", params=params, provider_config=provider_config)
    llm = bam.load()
    assert isinstance(llm, LangChainInterface)
    assert bam.default_params
    assert bam.params

    # taken from configuration
    assert bam.url == "test_url"
    assert bam.credentials == "secret_key"

    # known parameters should be there
    assert "min_new_tokens" in bam.params
    assert bam.params["min_new_tokens"] == 1

    assert "max_new_tokens" in bam.params
    assert bam.params["max_new_tokens"] == 10

    assert "temperature" in bam.params
    assert bam.params["temperature"] == 0.3

    # unknown parameters should be filtered out
    assert "verbose" not in bam.params

    assert "unknown_parameter" not in bam.params


def test_credentials_in_directory_handling(provider_config_credentials_directory):
    """Test that credentials in directory is handled as expected."""
    params = {}

    bam = BAM(
        model="uber-model",
        params=params,
        provider_config=provider_config_credentials_directory,
    )
    llm = bam.load()
    assert isinstance(llm, LangChainInterface)

    # taken from configuration
    assert bam.credentials == "secret_key"


def test_params_handling_specific_params(provider_config_with_specific_params):
    """Test that provider-specific parameters take precedence."""
    bam = BAM(
        model="uber-model",
        params={},
        provider_config=provider_config_with_specific_params,
    )
    llm = bam.load()
    assert isinstance(llm, LangChainInterface)
    assert bam.default_params
    assert bam.params

    # parameters taken from provier-specific configuration
    # which takes precedence over regular configuration
    assert bam.url == "http://bam.com/"
    assert bam.credentials == "secret_key_2"


def test_params_handling_none_values(provider_config):
    """Test handling parameters with None values."""
    # first two parameters should be removed before model init
    # rest need to stay
    params = {
        "unknown_parameter": None,
        "verbose": None,
        "min_new_tokens": None,
        "max_new_tokens": None,
        "temperature": None,
    }

    bam = BAM(model="uber-model", params=params, provider_config=provider_config)
    llm = bam.load()
    assert isinstance(llm, LangChainInterface)
    assert bam.default_params
    assert bam.params

    # known parameters should be there
    assert "min_new_tokens" in bam.params
    assert bam.params["min_new_tokens"] is None

    assert "max_new_tokens" in bam.params
    assert bam.params["max_new_tokens"] is None

    assert "temperature" in bam.params
    assert bam.params["temperature"] is None

    # unknown parameters should be filtered out
    assert "verbose" not in bam.params

    assert "unknown_parameter" not in bam.params


def test_params_replace_default_values_with_none(provider_config):
    """Test if default values are replaced by None values."""
    # provider initialization with empty set of params
    bam = BAM(model="uber-model", params={}, provider_config=provider_config)
    bam.load()

    # check default value
    assert "decoding_method" in bam.params
    assert bam.params["decoding_method"] == "sample"

    # provider initialization where default parameter is overriden
    params = {"decoding_method": None}

    bam = BAM(model="uber-model", params=params, provider_config=provider_config)
    bam.load()

    # check default value overrided by None
    assert "decoding_method" in bam.params
    assert bam.params["decoding_method"] is None


def test_missing_credentials_check(provider_config_without_credentials):
    """Test that check for missing credentials is in place ."""
    bam = BAM(
        model="uber-model",
        params={},
        provider_config=provider_config_without_credentials,
    )
    with pytest.raises(ValueError, match="Credentials must be specified"):
        bam.load()
