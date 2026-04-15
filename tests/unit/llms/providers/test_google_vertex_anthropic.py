"""Unit tests for Google Vertex AI provider."""

from unittest.mock import patch

import pytest

from ols.app.models.config import ProviderConfig
from ols.src.llms.providers.google_vertex_anthropic import GoogleVertexAnthropic

from .utils import generate_service_account_json_string


@pytest.fixture
def provider_config(tmpdir):
    """Fixture with provider configuration for Google Vertex AI."""
    credentials_json = generate_service_account_json_string()
    p = tmpdir.mkdir("sub").join("service-account.json")
    p.write(credentials_json)
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "google_vertex_anthropic",
            "url": "https://us-east5-aiplatform.googleapis.com",
            "credentials_path": p.strpath,
            "google_vertex_anthropic_config": {
                "project": "my-specific-project",
                "location": "us-east5",
            },
            "models": [
                {
                    "name": "claude-opus-4-6",
                }
            ],
        }
    )


@pytest.fixture
def provider_config_with_specific_parameters(tmpdir):
    """Fixture with provider configuration for Google Vertex AI with specific parameters."""
    credentials_json = generate_service_account_json_string()
    p = tmpdir.mkdir("sub").join("service-account.json")
    p.write(credentials_json)
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "google_vertex_anthropic",
            "url": "https://europe-west1-aiplatform.googleapis.com",
            "credentials_path": p.strpath,
            "google_vertex_anthropic_config": {
                "project": "my-specific-project",
                "location": "europe-west1",
            },
            "models": [
                {
                    "name": "claude-opus-4-6",
                }
            ],
        }
    )


@patch(
    "ols.src.llms.providers.google_vertex_anthropic.ChatAnthropicVertex",
    autospec=True,
)
def test_basic_interface(mock_chat, provider_config):
    """Test basic interface."""
    vertex = GoogleVertexAnthropic(
        model="claude-opus-4-6", params={}, provider_config=provider_config
    )
    llm = vertex.load()
    assert llm is not None
    assert vertex.default_params
    assert "model_name" in vertex.default_params
    assert "project" in vertex.default_params
    assert "location" in vertex.default_params
    assert "max_output_tokens" in vertex.default_params
    assert vertex.default_params["project"] == "my-specific-project"
    assert vertex.default_params["location"] == "us-east5"

    mock_chat.assert_called_once()
    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["project"] == "my-specific-project"
    assert call_kwargs["location"] == "us-east5"
    assert call_kwargs["model_name"] == "claude-opus-4-6"


@patch(
    "ols.src.llms.providers.google_vertex_anthropic.ChatAnthropicVertex",
    autospec=True,
)
def test_params_handling(mock_chat, provider_config):
    """Test that not allowed parameters are removed before model init."""
    params = {
        "unknown_parameter": "foo",
        "min_new_tokens": 1,
        "max_new_tokens": 10,
        "temperature": 0.3,
    }

    vertex = GoogleVertexAnthropic(
        model="claude-opus-4-6", params=params, provider_config=provider_config
    )
    llm = vertex.load()
    assert llm is not None
    assert vertex.default_params
    assert vertex.params

    assert "temperature" in vertex.params
    assert vertex.params["temperature"] == 0.3

    assert "min_new_tokens" not in vertex.params
    assert "max_new_tokens" not in vertex.params
    assert "unknown_parameter" not in vertex.params


@patch(
    "ols.src.llms.providers.google_vertex_anthropic.ChatAnthropicVertex",
    autospec=True,
)
def test_loading_provider_specific_parameters(
    mock_chat, provider_config_with_specific_parameters
):
    """Test that provider-specific config overrides generic config."""
    vertex = GoogleVertexAnthropic(
        model="claude-opus-4-6",
        params={},
        provider_config=provider_config_with_specific_parameters,
    )
    llm = vertex.load()
    assert llm is not None
    assert vertex.default_params
    assert vertex.params

    assert vertex.project == "my-specific-project"
    assert vertex.location == "europe-west1"
    assert vertex.default_params["project"] == "my-specific-project"
    assert vertex.default_params["location"] == "europe-west1"

    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["project"] == "my-specific-project"
    assert call_kwargs["location"] == "europe-west1"
