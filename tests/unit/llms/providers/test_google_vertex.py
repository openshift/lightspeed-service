"""Unit tests for Google Vertex AI provider (ChatGoogleGenerativeAI on Vertex)."""

from unittest.mock import patch

import pytest

from ols.app.models.config import ProviderConfig
from ols.src.llms.providers.google_vertex import GoogleVertex

from .utils import generate_service_account_json_string


@pytest.fixture
def provider_config(tmpdir):
    """Fixture with provider configuration for Vertex."""
    credentials_json = generate_service_account_json_string()
    p = tmpdir.mkdir("sub").join("service-account.json")
    p.write(credentials_json)
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "google_vertex",
            "url": "https://us-central1-aiplatform.googleapis.com",
            "credentials_path": p.strpath,
            "project_id": "my-gcp-project",
            "models": [
                {
                    "name": "gemini-2.5-flash",
                }
            ],
        }
    )


@pytest.fixture
def provider_config_with_specific_parameters(tmpdir):
    """Fixture with Vertex and explicit google_vertex_config."""
    credentials_json = generate_service_account_json_string()
    p = tmpdir.mkdir("sub").join("service-account.json")
    p.write(credentials_json)
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "google_vertex",
            "url": "https://us-central1-aiplatform.googleapis.com",
            "credentials_path": p.strpath,
            "google_vertex_config": {
                "project": "my-specific-project",
                "location": "us-central1",
            },
            "models": [
                {
                    "name": "gemini-2.5-flash",
                }
            ],
        }
    )


@patch(
    "ols.src.llms.providers.google_vertex.ChatGoogleGenerativeAI",
    autospec=True,
)
def test_basic_interface(mock_chat, provider_config):
    """Test basic interface."""
    vertex = GoogleVertex(
        model="gemini-2.5-flash", params={}, provider_config=provider_config
    )
    llm = vertex.load()
    assert llm is not None
    assert vertex.default_params
    assert "model" in vertex.default_params
    assert "project" in vertex.default_params
    assert "location" in vertex.default_params
    assert "max_output_tokens" in vertex.default_params
    assert vertex.default_params["project"] == "my-gcp-project"
    assert vertex.default_params["location"] == "global"
    assert vertex.default_params["vertexai"] is True

    mock_chat.assert_called_once()
    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["project"] == "my-gcp-project"
    assert call_kwargs["location"] == "global"
    assert call_kwargs["model"] == "gemini-2.5-flash"
    assert call_kwargs["vertexai"] is True


@patch(
    "ols.src.llms.providers.google_vertex.ChatGoogleGenerativeAI",
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

    vertex = GoogleVertex(
        model="gemini-2.5-flash", params=params, provider_config=provider_config
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
    "ols.src.llms.providers.google_vertex.ChatGoogleGenerativeAI",
    autospec=True,
)
def test_loading_provider_specific_parameters(
    mock_chat, provider_config_with_specific_parameters
):
    """Test that provider-specific config overrides generic config."""
    vertex = GoogleVertex(
        model="gemini-2.5-flash",
        params={},
        provider_config=provider_config_with_specific_parameters,
    )
    llm = vertex.load()
    assert llm is not None
    assert vertex.default_params
    assert vertex.params

    assert vertex.project == "my-specific-project"
    assert vertex.location == "us-central1"
    assert vertex.default_params["project"] == "my-specific-project"
    assert vertex.default_params["location"] == "us-central1"

    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["project"] == "my-specific-project"
    assert call_kwargs["location"] == "us-central1"
