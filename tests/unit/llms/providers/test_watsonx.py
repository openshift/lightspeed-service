"""Unit tests for Watsonx provider."""

from unittest.mock import patch

import pytest
from ibm_watson_machine_learning.metanames import (
    GenTextParamsMetaNames as GenParams,
)

from ols.app.models.config import ProviderConfig
from ols.constants import GenericLLMParameters
from ols.src.llms.providers.watsonx import Watsonx
from tests.mock_classes.mock_watsonxllm import WatsonxLLM


@pytest.fixture
def provider_config():
    """Fixture with provider configuration for Watsonx."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "watsonx",
            "url": "https://us-south.ml.cloud.ibm.com",
            "credentials_path": "tests/config/secret/apitoken",
            "project_id": "01234567-89ab-cdef-0123-456789abcdef",
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
    """Fixture with provider configuration for Watsonx."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "watsonx",
            "url": "https://us-south.ml.cloud.ibm.com",
            "credentials_path": "tests/config/secret",
            "project_id": "01234567-89ab-cdef-0123-456789abcdef",
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
def provider_config_without_credentials():
    """Fixture with provider configuration for Watsonx without credentials."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "watsonx",
            "url": "https://us-south.ml.cloud.ibm.com",
            "project_id": "01234567-89ab-cdef-0123-456789abcdef",
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
def provider_config_with_specific_params():
    """Fixture with provider configuration for Watsonx with provider-specific parameters."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "watsonx",
            "url": "https://us-south.ml.cloud.ibm.com",
            "credentials_path": "tests/config/secret/apitoken",
            "project_id": "01234567-89ab-cdef-0123-456789abcdef",
            "watsonx_config": {
                "url": "http://bam.com",
                "credentials_path": "tests/config/secret2/apitoken",
                "project_id": "ffffffff-89ab-cdef-0123-456789abcdef",
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


@patch("ols.src.llms.providers.watsonx.WatsonxLLM", new=WatsonxLLM())
def test_basic_interface(provider_config):
    """Test basic interface."""
    watsonx = Watsonx(model="uber-model", params={}, provider_config=provider_config)
    llm = watsonx.load()
    assert isinstance(llm, WatsonxLLM)
    assert watsonx.default_params


@patch("ols.src.llms.providers.watsonx.WatsonxLLM", new=WatsonxLLM())
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

    watsonx = Watsonx(
        model="uber-model", params=params, provider_config=provider_config
    )
    llm = watsonx.load()
    assert isinstance(llm, WatsonxLLM)
    assert watsonx.default_params
    assert watsonx.params

    # taken from configuration
    assert watsonx.url == "https://us-south.ml.cloud.ibm.com"
    assert watsonx.credentials == "secret_key"
    assert watsonx.project_id == "01234567-89ab-cdef-0123-456789abcdef"

    # known parameters should be there
    assert GenParams.DECODING_METHOD in watsonx.params
    assert watsonx.params[GenParams.DECODING_METHOD] == "sample"

    assert GenParams.MAX_NEW_TOKENS in watsonx.params
    assert watsonx.params[GenParams.MAX_NEW_TOKENS] == 10

    # unknown parameters should be filtered out
    assert "unknown_parameter" not in watsonx.params
    assert "verbose" not in watsonx.params


@patch("ols.src.llms.providers.watsonx.WatsonxLLM", new=WatsonxLLM())
def test_credentials_key_in_directory_handling(provider_config_credentials_directory):
    """Test that credentials in directory is handled as expected."""
    params = {}

    watsonx = Watsonx(
        model="uber-model",
        params=params,
        provider_config=provider_config_credentials_directory,
    )
    llm = watsonx.load()
    assert isinstance(llm, WatsonxLLM)

    # taken from configuration
    assert watsonx.credentials == "secret_key"


@patch("ols.src.llms.providers.watsonx.WatsonxLLM", new=WatsonxLLM())
def test_params_handling_specific_params(provider_config_with_specific_params):
    """Test that provider-specific parameters take precedence."""
    watsonx = Watsonx(
        model="uber-model",
        params={},
        provider_config=provider_config_with_specific_params,
    )
    llm = watsonx.load()
    assert isinstance(llm, WatsonxLLM)
    assert watsonx.default_params
    assert watsonx.params

    # parameters taken from provier-specific configuration
    # which takes precedence over regular configuration
    assert watsonx.url == "http://bam.com/"
    assert watsonx.credentials == "secret_key_2"
    assert watsonx.project_id == "ffffffff-89ab-cdef-0123-456789abcdef"


@patch("ols.src.llms.providers.watsonx.WatsonxLLM", new=WatsonxLLM())
def test_params_handling_none_values(provider_config):
    """Test handling parameters with None values."""
    # first three parameters should be removed before model init
    # rest need to stay
    params = {
        "unknown_parameter": None,
        "temperature": None,
        "verbose": None,
        "min_new_tokens": None,
        "max_new_tokens": None,
    }

    watsonx = Watsonx(
        model="uber-model", params=params, provider_config=provider_config
    )
    llm = watsonx.load()
    assert isinstance(llm, WatsonxLLM)
    assert watsonx.default_params
    assert watsonx.params

    # known parameters should be there
    assert GenParams.MIN_NEW_TOKENS in watsonx.params
    assert watsonx.params[GenParams.MIN_NEW_TOKENS] is None

    assert GenParams.MAX_NEW_TOKENS in watsonx.params
    assert watsonx.params[GenParams.MAX_NEW_TOKENS] is None

    assert GenParams.TEMPERATURE in watsonx.params
    assert watsonx.params[GenParams.TEMPERATURE] is None

    # unknown parameters should be filtered out
    assert "unknown_parameter" not in watsonx.params
    assert "verbose" not in watsonx.params


@patch("ols.src.llms.providers.watsonx.WatsonxLLM", new=WatsonxLLM())
def test_params_replace_default_values_with_none(provider_config):
    """Test if default values are replaced by None values."""
    # provider initialization with empty set of params
    watsonx = Watsonx(model="uber-model", params={}, provider_config=provider_config)
    watsonx.load()

    # check default value
    assert GenParams.DECODING_METHOD in watsonx.params
    assert watsonx.params[GenParams.DECODING_METHOD] == "sample"

    # provider initialization where default parameter is overriden
    params = {"decoding_method": None}

    watsonx = Watsonx(
        model="uber-model", params=params, provider_config=provider_config
    )
    watsonx.load()

    # check default value overrided by None
    assert GenParams.DECODING_METHOD in watsonx.params
    assert watsonx.params[GenParams.DECODING_METHOD] is None


@patch("ols.src.llms.providers.watsonx.WatsonxLLM", new=WatsonxLLM())
def test_generic_parameter_mappings(provider_config):
    """Test generic parameter mapping to provider parameter list."""
    # some non-default values for generic LLM parameters
    generic_llm_params = {
        GenericLLMParameters.MIN_TOKENS_FOR_RESPONSE: 100,
        GenericLLMParameters.MAX_TOKENS_FOR_RESPONSE: 200,
        GenericLLMParameters.TOP_K: 10,
        GenericLLMParameters.TOP_P: 1.5,
        GenericLLMParameters.TEMPERATURE: 42.0,
    }

    watsonx = Watsonx(
        model="uber-model", params=generic_llm_params, provider_config=provider_config
    )
    llm = watsonx.load()
    assert isinstance(llm, WatsonxLLM)
    assert watsonx.default_params
    assert watsonx.params

    # generic parameters should be remapped to Watsonx-specific parameters
    assert GenParams.MIN_NEW_TOKENS in watsonx.params
    assert GenParams.MAX_NEW_TOKENS in watsonx.params
    assert GenParams.TOP_K in watsonx.params
    assert GenParams.TOP_P in watsonx.params
    assert GenParams.TEMPERATURE in watsonx.params
    assert watsonx.params[GenParams.MIN_NEW_TOKENS] == 100
    assert watsonx.params[GenParams.MAX_NEW_TOKENS] == 200
    assert watsonx.params[GenParams.TOP_K] == 10
    assert watsonx.params[GenParams.TOP_P] == 1.5
    assert watsonx.params[GenParams.TEMPERATURE] == 42.0


def test_missing_credentials_check(provider_config_without_credentials):
    """Test that check for missing credentials is in place ."""
    watsonx = Watsonx(
        model="uber-model",
        params={},
        provider_config=provider_config_without_credentials,
    )
    with pytest.raises(ValueError, match="Credentials must be specified"):
        watsonx.load()
