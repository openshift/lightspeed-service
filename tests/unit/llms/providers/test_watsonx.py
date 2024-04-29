"""Unit tests for Watsonx provider."""

from unittest.mock import patch

import pytest
from ibm_watson_machine_learning.metanames import (
    GenTextParamsMetaNames as GenParams,
)

from ols.app.models.config import ProviderConfig
from ols.src.llms.providers.watsonx import WatsonX
from ols.utils import config
from tests.mock_classes.mock_watsonxllm import WatsonxLLM


@pytest.fixture
def provider_config():
    """Fixture with provider configuration for Watsonx."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "watsonx",
            "url": "https://us-south.ml.cloud.ibm.com",
            "credentials_path": "tests/config/secret.txt",
            "project_id": "01234567-89ab-cdef-0123-456789abcdef",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "test_model_url",
                    "credentials_path": "tests/config/secret.txt",
                }
            ],
        }
    )


@patch("ols.src.llms.providers.watsonx.WatsonxLLM", new=WatsonxLLM())
def test_basic_interface(provider_config):
    """Test basic interface."""
    config.init_empty_config()  # needed for checking the config.dev_config.llm_params

    watsonx = WatsonX(model="uber-model", params={}, provider_config=provider_config)
    llm = watsonx.load()
    assert isinstance(llm, WatsonxLLM)
    assert watsonx.default_params


@patch("ols.src.llms.providers.watsonx.WatsonxLLM", new=WatsonxLLM())
def test_params_handling(provider_config):
    """Test that not allowed parameters are removed before model init."""
    config.init_empty_config()  # needed for checking the config.dev_config.llm_params

    # first two parameters should be removed before model init
    # rest need to stay
    params = {
        "unknown_parameter": "foo",
        "verbose": True,
        "min_new_tokens": 1,
        "max_new_tokens": 10,
        "temperature": 0.3,
    }

    watsonx = WatsonX(
        model="uber-model", params=params, provider_config=provider_config
    )
    llm = watsonx.load()
    assert isinstance(llm, WatsonxLLM)
    assert watsonx.default_params
    assert watsonx.params

    # known parameters should be there
    assert GenParams.DECODING_METHOD in watsonx.params
    assert watsonx.params[GenParams.DECODING_METHOD] == "sample"

    assert GenParams.MAX_NEW_TOKENS in watsonx.params
    assert watsonx.params[GenParams.MAX_NEW_TOKENS] == 10

    # unknown parameters should be filtered out
    assert "unknown_parameter" not in watsonx.params
    assert "verbose" not in watsonx.params


@patch("ols.src.llms.providers.watsonx.WatsonxLLM", new=WatsonxLLM())
def test_params_handling_none_values(provider_config):
    """Test handling parameters with None values."""
    config.init_empty_config()  # needed for checking the config.dev_config.llm_params

    # first three parameters should be removed before model init
    # rest need to stay
    params = {
        "unknown_parameter": None,
        "temperature": None,
        "verbose": None,
        "min_new_tokens": None,
        "max_new_tokens": None,
    }

    watsonx = WatsonX(
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
    config.init_empty_config()  # needed for checking the config.dev_config.llm_params

    # provider initialization with empty set of params
    watsonx = WatsonX(model="uber-model", params={}, provider_config=provider_config)
    watsonx.load()

    # check default value
    assert GenParams.DECODING_METHOD in watsonx.params
    assert watsonx.params[GenParams.DECODING_METHOD] == "sample"

    # provider initialization where default parameter is overriden
    params = {"decoding_method": None}

    watsonx = WatsonX(
        model="uber-model", params=params, provider_config=provider_config
    )
    watsonx.load()

    # check default value overrided by None
    assert GenParams.DECODING_METHOD in watsonx.params
    assert watsonx.params[GenParams.DECODING_METHOD] is None
