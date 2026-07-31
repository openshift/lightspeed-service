"""Unit tests for Anthropic provider."""

import os
from unittest.mock import patch

import httpx
import pytest

from ols import constants
from ols.app.models.config import ProviderConfig
from ols.src.llms.providers.anthropic import Anthropic

cert_in_certificates_store_path = "tests/unit/extra_certs/sample_cert_1.crt"


@pytest.fixture
def fake_certifi_store(tmpdir):
    """Create a fake certifi store."""
    cert_store_path = os.path.join(
        constants.DEFAULT_CERTIFICATE_DIRECTORY, constants.CERTIFICATE_STORAGE_FILENAME
    )
    with open(cert_store_path, "wb") as cert_store:
        with open(cert_in_certificates_store_path, "rb") as cert_file:
            cert_store.write(cert_file.read())
    return cert_store_path


@pytest.fixture
def provider_config():
    """Fixture with provider configuration for Anthropic."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "anthropic",
            "url": "https://api.anthropic.com",
            "credentials_path": "tests/config/secret/apitoken",
            "models": [
                {
                    "name": "claude-sonnet-4-20250514",
                    "url": "https://api.anthropic.com",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )


@pytest.fixture
def provider_config_with_specific_parameters():
    """Fixture with Anthropic provider-specific config."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "anthropic",
            "url": "https://api.anthropic.com",
            "credentials_path": "tests/config/secret/apitoken",
            "anthropic_config": {
                "url": "https://custom-anthropic.example.com",
                "credentials_path": "tests/config/secret2/apitoken",
            },
            "models": [
                {
                    "name": "claude-sonnet-4-20250514",
                    "url": "https://api.anthropic.com",
                    "credentials_path": "tests/config/secret/apitoken",
                }
            ],
        }
    )


@pytest.fixture
def provider_config_with_thinking():
    """Fixture with Anthropic provider config with thinking enabled."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "anthropic",
            "url": "https://api.anthropic.com",
            "credentials_path": "tests/config/secret/apitoken",
            "models": [
                {
                    "name": "claude-sonnet-4-20250514",
                    "url": "https://api.anthropic.com",
                    "credentials_path": "tests/config/secret/apitoken",
                    "options": {
                        "reasoning_config": {
                            "type": "enabled",
                            "budget_tokens": 10000,
                        }
                    },
                }
            ],
        }
    )


@patch(
    "ols.src.llms.providers.anthropic.ChatAnthropic",
    autospec=True,
)
def test_basic_interface(mock_chat_anthropic, provider_config, fake_certifi_store):
    """Test basic interface."""
    anthropic = Anthropic(
        model="claude-sonnet-4-20250514", params={}, provider_config=provider_config
    )
    llm = anthropic.load()
    assert llm is not None
    assert anthropic.default_params
    assert "base_url" in anthropic.default_params
    assert "model" in anthropic.default_params
    assert "max_tokens" in anthropic.default_params
    assert "anthropic_api_key" in anthropic.default_params

    assert "http_client" in anthropic.default_params
    assert anthropic.default_params["http_client"] is not None
    assert "http_async_client" in anthropic.default_params
    assert anthropic.default_params["http_async_client"] is not None

    client = anthropic.default_params["http_client"]
    assert isinstance(client, httpx.Client)


@patch(
    "ols.src.llms.providers.anthropic.ChatAnthropic",
    autospec=True,
)
def test_params_handling(mock_chat_anthropic, provider_config, fake_certifi_store):
    """Test that not allowed parameters are removed before model init."""
    params = {
        "unknown_parameter": "foo",
        "min_new_tokens": 1,
        "max_new_tokens": 10,
        "temperature": 0.3,
        "top_k": 5,
    }

    anthropic = Anthropic(
        model="claude-sonnet-4-20250514",
        params=params,
        provider_config=provider_config,
    )
    anthropic.load()
    assert anthropic.default_params
    assert anthropic.params

    # known parameters should be there
    assert "temperature" in anthropic.params
    assert "top_k" in anthropic.params
    assert anthropic.params["temperature"] == 0.3
    assert anthropic.params["top_k"] == 5

    # unknown parameters should be filtered out
    assert "min_new_tokens" not in anthropic.params
    assert "max_new_tokens" not in anthropic.params
    assert "unknown_parameter" not in anthropic.params

    # taken from configuration
    assert anthropic.url == "https://api.anthropic.com"
    assert anthropic.credentials == "secret_key"

    assert anthropic.default_params["anthropic_api_key"] == "secret_key"
    assert anthropic.default_params["base_url"] == "https://api.anthropic.com"


@patch(
    "ols.src.llms.providers.anthropic.ChatAnthropic",
    autospec=True,
)
def test_loading_provider_specific_parameters(
    mock_chat_anthropic, provider_config_with_specific_parameters, fake_certifi_store
):
    """Test that provider-specific config takes precedence."""
    anthropic = Anthropic(
        model="claude-sonnet-4-20250514",
        params={},
        provider_config=provider_config_with_specific_parameters,
    )
    anthropic.load()
    assert anthropic.default_params
    assert anthropic.params

    assert "base_url" in anthropic.default_params
    assert "model" in anthropic.default_params
    assert "max_tokens" in anthropic.default_params

    # parameters taken from provider-specific configuration
    # which takes precedence over regular configuration
    assert anthropic.url == "https://custom-anthropic.example.com/"
    assert anthropic.credentials == "secret_key_2"

    assert anthropic.default_params["anthropic_api_key"] == "secret_key_2"
    assert (
        anthropic.default_params["base_url"] == "https://custom-anthropic.example.com/"
    )


@patch(
    "ols.src.llms.providers.anthropic.ChatAnthropic",
    autospec=True,
)
def test_none_params_handling(mock_chat_anthropic, provider_config, fake_certifi_store):
    """Test that None-valued known parameters are kept."""
    params = {
        "unknown_parameter": None,
        "min_new_tokens": None,
        "max_new_tokens": None,
        "model": None,
        "base_url": None,
    }

    anthropic = Anthropic(
        model="claude-sonnet-4-20250514",
        params=params,
        provider_config=provider_config,
    )
    anthropic.load()
    assert anthropic.default_params
    assert anthropic.params

    assert anthropic.default_params["anthropic_api_key"] == "secret_key"
    assert anthropic.default_params["base_url"] == "https://api.anthropic.com"


@patch(
    "ols.src.llms.providers.anthropic.ChatAnthropic",
    autospec=True,
)
def test_thinking_config(
    mock_chat_anthropic, provider_config_with_thinking, fake_certifi_store
):
    """Test that reasoning_config in options produces thinking param."""
    anthropic = Anthropic(
        model="claude-sonnet-4-20250514",
        params={},
        provider_config=provider_config_with_thinking,
    )
    anthropic.load()

    # thinking should be set from reasoning_config
    assert "thinking" in anthropic.default_params
    assert anthropic.default_params["thinking"]["type"] == "enabled"
    assert anthropic.default_params["thinking"]["budget_tokens"] == 10000

    # temperature and top_p should be removed when thinking is enabled
    assert "temperature" not in anthropic.default_params
    assert "top_p" not in anthropic.default_params

    # core parameters should still be present
    assert "model" in anthropic.default_params
    assert "anthropic_api_key" in anthropic.default_params
    assert "base_url" in anthropic.default_params
    assert "max_tokens" in anthropic.default_params
