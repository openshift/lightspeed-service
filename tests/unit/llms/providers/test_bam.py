"""Unit tests for BAM provider."""

import pytest
from genai.extensions.langchain import LangChainInterface

from ols import config
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
            "credentials_path": "tests/config/secret.txt",
            "models": [
                {
                    "name": "test_model_name",
                    "url": "test_model_url",
                    "credentials_path": "tests/config/secret.txt",
                }
            ],
        }
    )


def test_basic_interface(provider_config):
    """Test basic interface."""
    config.reload_empty()  # needed for checking the config.dev_config.llm_params

    bam = BAM(model="uber-model", params={}, provider_config=provider_config)
    llm = bam.load()
    assert isinstance(llm, LangChainInterface)
    assert bam.default_params


def test_params_handling(provider_config):
    """Test that not allowed parameters are removed before model init."""
    config.reload_empty()  # needed for checking the config.dev_config.llm_params

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


def test_params_handling_none_values(provider_config):
    """Test handling parameters with None values."""
    config.reload_empty()  # needed for checking the config.dev_config.llm_params

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
    config.reload_empty()  # needed for checking the config.dev_config.llm_params

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
