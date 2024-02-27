"""Unit tests for OpenAI provider."""

import pytest
from langchain_openai.chat_models.base import ChatOpenAI

from ols.app.models.config import ProviderConfig
from ols.src.llms.providers.openai import OpenAI
from ols.utils import config


@pytest.fixture
def provider_config():
    """Fixture with provider configuration for OpenAI."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "openai",
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
    config.init_empty_config()  # needed for checking the config.dev_config.llm_params

    openai = OpenAI(model="uber-model", params={}, provider_config=provider_config)
    llm = openai.load()
    assert isinstance(llm, ChatOpenAI)
    assert openai.default_params
    assert "base_url" in openai.default_params
    assert "model" in openai.default_params
    assert "max_tokens" in openai.default_params


def test_params_handling(provider_config):
    """Test that not allowed parameters are removed before model init."""
    config.init_empty_config()  # needed for checking the config.dev_config.llm_params

    # these two parameters should be removed before model init
    params = {
        "min_new_tokens": 1,
        "max_new_tokens": 10,
    }

    openai = OpenAI(model="uber-model", params=params, provider_config=provider_config)
    llm = openai.load()
    assert isinstance(llm, ChatOpenAI)
    assert openai.default_params
    assert "min_new_tokens" not in openai.default_params
    assert "max_new_tokens" not in openai.default_params
