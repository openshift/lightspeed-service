"""Unit tests for Google Vertex AI providers (Gemini and Anthropic on Vertex)."""

import json
from unittest.mock import patch

import pytest

from ols.app.models.config import ProviderConfig
from ols.src.llms.providers.google_vertex import GoogleVertex, GoogleVertexAnthropic

from .utils import generate_service_account_json_string


@pytest.fixture
def gemini_provider_config(tmpdir):
    """Return provider configuration for Vertex Gemini."""
    credentials_json = generate_service_account_json_string()
    p = tmpdir.mkdir("sub").join("service-account.json")
    p.write(credentials_json)
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "google_vertex",
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
def gemini_provider_config_with_specific_parameters(tmpdir):
    """Return Gemini Vertex config with explicit google_vertex_config."""
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


@pytest.fixture
def gemini_provider_config_authorized_user(tmpdir):
    """Return Vertex Gemini config with authorized_user credentials JSON."""
    credentials_json = json.dumps(
        {
            "type": "authorized_user",
            "client_id": "authorized-user-client-id",
            "client_secret": "authorized-user-client-secret",
            "refresh_token": "authorized-user-refresh-token",
        }
    )
    p = tmpdir.mkdir("sub").join("authorized-user.json")
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


@pytest.fixture
def anthropic_provider_config(tmpdir):
    """Return provider configuration for Vertex Anthropic."""
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
def anthropic_provider_config_with_specific_parameters(tmpdir):
    """Return Vertex Anthropic config with alternate region."""
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


@pytest.fixture
def anthropic_provider_config_authorized_user(tmpdir):
    """Return Vertex Anthropic config with authorized_user credentials JSON."""
    credentials_json = json.dumps(
        {
            "type": "authorized_user",
            "client_id": "authorized-user-client-id",
            "client_secret": "authorized-user-client-secret",
            "refresh_token": "authorized-user-refresh-token",
        }
    )
    p = tmpdir.mkdir("sub").join("authorized-user.json")
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


@patch(
    "ols.src.llms.providers.google_vertex.ChatGoogleGenerativeAI",
    autospec=True,
)
def test_gemini_basic_interface(mock_chat, gemini_provider_config):
    """Test Gemini Vertex basic interface."""
    vertex = GoogleVertex(
        model="gemini-2.5-flash", params={}, provider_config=gemini_provider_config
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
    assert "base_url" not in call_kwargs


@pytest.mark.parametrize("temperature", [None, 0.0, 0.7])
@patch("ols.src.llms.providers.google_vertex.ChatGoogleGenerativeAI", autospec=True)
def test_gemini_uses_configured_model_temperature(
    mock_chat, gemini_provider_config: ProviderConfig, temperature: float | None
) -> None:
    """Send only configured temperatures to Gemini on Vertex."""
    gemini_provider_config.models["gemini-2.5-flash"].parameters.temperature = (
        temperature
    )
    GoogleVertex(
        model="gemini-2.5-flash", provider_config=gemini_provider_config
    ).load()
    kwargs = mock_chat.call_args.kwargs
    if temperature is None:
        assert "temperature" not in kwargs
    else:
        assert kwargs["temperature"] == temperature


@patch("ols.src.llms.providers.google_vertex.ChatGoogleGenerativeAI", autospec=True)
def test_gemini_params_handling(mock_chat, gemini_provider_config):
    """Test Gemini Vertex strips disallowed parameters before model init."""
    params = {
        "unknown_parameter": "foo",
        "min_new_tokens": 1,
        "max_new_tokens": 10,
        "temperature": 0.3,
    }

    vertex = GoogleVertex(
        model="gemini-2.5-flash", params=params, provider_config=gemini_provider_config
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
def test_gemini_loading_provider_specific_parameters(
    mock_chat, gemini_provider_config_with_specific_parameters
):
    """Test Gemini Vertex google_vertex_config overrides project and location."""
    vertex = GoogleVertex(
        model="gemini-2.5-flash",
        params={},
        provider_config=gemini_provider_config_with_specific_parameters,
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
    assert call_kwargs["base_url"] == "https://us-central1-aiplatform.googleapis.com"


@patch(
    "ols.src.llms.providers.google_vertex.ChatGoogleGenerativeAI",
    autospec=True,
)
def test_gemini_authorized_user_credentials(
    mock_chat, gemini_provider_config_authorized_user
):
    """Test Gemini on Vertex accepts authorized_user credentials JSON."""
    vertex = GoogleVertex(
        model="gemini-2.5-flash",
        params={},
        provider_config=gemini_provider_config_authorized_user,
    )
    llm = vertex.load()
    assert llm is not None
    assert vertex.default_params

    payload = json.loads(gemini_provider_config_authorized_user.credentials)
    expected_refresh_token = payload["refresh_token"]
    default_credentials = vertex.default_params["credentials"]
    assert default_credentials.refresh_token == expected_refresh_token

    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["project"] == "my-specific-project"
    assert call_kwargs["location"] == "us-central1"
    assert call_kwargs["model"] == "gemini-2.5-flash"
    assert call_kwargs["vertexai"] is True
    assert call_kwargs["base_url"] == "https://us-central1-aiplatform.googleapis.com"


@pytest.mark.parametrize("temperature", [None, 0.0, 0.7])
@patch("ols.src.llms.providers.google_vertex.ChatAnthropicVertex", autospec=True)
def test_anthropic_uses_configured_model_temperature(
    mock_chat, anthropic_provider_config: ProviderConfig, temperature: float | None
) -> None:
    """Send only configured temperatures to Anthropic on Vertex."""
    anthropic_provider_config.models["claude-opus-4-6"].parameters.temperature = (
        temperature
    )
    GoogleVertexAnthropic(
        model="claude-opus-4-6", provider_config=anthropic_provider_config
    ).load()
    kwargs = mock_chat.call_args.kwargs
    if temperature is None:
        assert "temperature" not in kwargs
    else:
        assert kwargs["temperature"] == temperature


@patch("ols.src.llms.providers.google_vertex.ChatAnthropicVertex", autospec=True)
def test_anthropic_basic_interface(mock_chat, anthropic_provider_config):
    """Test Anthropic on Vertex basic interface."""
    vertex = GoogleVertexAnthropic(
        model="claude-opus-4-6", params={}, provider_config=anthropic_provider_config
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
    "ols.src.llms.providers.google_vertex.ChatAnthropicVertex",
    autospec=True,
)
def test_anthropic_params_handling(mock_chat, anthropic_provider_config):
    """Test Anthropic on Vertex strips disallowed parameters before model init."""
    params = {
        "unknown_parameter": "foo",
        "min_new_tokens": 1,
        "max_new_tokens": 10,
        "temperature": 0.3,
    }

    vertex = GoogleVertexAnthropic(
        model="claude-opus-4-6",
        params=params,
        provider_config=anthropic_provider_config,
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
    "ols.src.llms.providers.google_vertex.ChatAnthropicVertex",
    autospec=True,
)
def test_anthropic_loading_provider_specific_parameters(
    mock_chat, anthropic_provider_config_with_specific_parameters
):
    """Test Anthropic on Vertex config overrides region."""
    vertex = GoogleVertexAnthropic(
        model="claude-opus-4-6",
        params={},
        provider_config=anthropic_provider_config_with_specific_parameters,
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


@patch(
    "ols.src.llms.providers.google_vertex.ChatAnthropicVertex",
    autospec=True,
)
def test_anthropic_authorized_user_credentials(
    mock_chat, anthropic_provider_config_authorized_user
):
    """Test Anthropic on Vertex accepts authorized_user credentials JSON."""
    vertex = GoogleVertexAnthropic(
        model="claude-opus-4-6",
        params={},
        provider_config=anthropic_provider_config_authorized_user,
    )
    llm = vertex.load()
    assert llm is not None
    assert vertex.default_params

    payload = json.loads(anthropic_provider_config_authorized_user.credentials)
    expected_refresh_token = payload["refresh_token"]
    default_credentials = vertex.default_params["credentials"]
    assert default_credentials.refresh_token == expected_refresh_token

    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["project"] == "my-specific-project"
    assert call_kwargs["location"] == "us-east5"


@patch(
    "ols.src.llms.providers.google_vertex.ChatGoogleGenerativeAI",
    autospec=True,
)
def test_gemini_reasoning_config_with_include_thoughts_true(mock_chat, tmpdir):
    """Test Gemini respects include_thoughts=true in reasoning_config."""
    credentials_json = generate_service_account_json_string()
    p = tmpdir.mkdir("sub").join("service-account.json")
    p.write(credentials_json)

    config_dict = {
        "name": "some_provider",
        "type": "google_vertex",
        "credentials_path": p.strpath,
        "project_id": "my-gcp-project",
        "models": [
            {
                "name": "gemini-2.5-flash",
                "parameters": {
                    "reasoning_config": {
                        "include_thoughts": True,
                        "thinking_level": "high",
                        "thinking_budget": 5000,
                    }
                },
            }
        ],
    }
    provider_config = ProviderConfig(config_dict)
    vertex = GoogleVertex(
        model="gemini-2.5-flash", params={}, provider_config=provider_config
    )
    llm = vertex.load()
    assert llm is not None
    assert vertex.default_params["include_thoughts"] is True
    assert vertex.default_params["thinking_level"] == "high"
    assert vertex.default_params["thinking_budget"] == 5000

    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["include_thoughts"] is True
    assert call_kwargs["thinking_level"] == "high"
    assert call_kwargs["thinking_budget"] == 5000


@patch(
    "ols.src.llms.providers.google_vertex.ChatGoogleGenerativeAI",
    autospec=True,
)
def test_gemini_reasoning_config_with_include_thoughts_false(mock_chat, tmpdir):
    """Test Gemini respects include_thoughts=false in reasoning_config."""
    credentials_json = generate_service_account_json_string()
    p = tmpdir.mkdir("sub").join("service-account.json")
    p.write(credentials_json)

    config_dict = {
        "name": "some_provider",
        "type": "google_vertex",
        "credentials_path": p.strpath,
        "project_id": "my-gcp-project",
        "models": [
            {
                "name": "gemini-2.5-flash",
                "parameters": {
                    "reasoning_config": {
                        "include_thoughts": False,
                        "thinking_level": "low",
                    }
                },
            }
        ],
    }
    provider_config = ProviderConfig(config_dict)
    vertex = GoogleVertex(
        model="gemini-2.5-flash", params={}, provider_config=provider_config
    )
    llm = vertex.load()
    assert llm is not None
    assert vertex.default_params["include_thoughts"] is False
    assert vertex.default_params["thinking_level"] == "low"

    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["include_thoughts"] is False
    assert call_kwargs["thinking_level"] == "low"


@patch(
    "ols.src.llms.providers.google_vertex.ChatGoogleGenerativeAI",
    autospec=True,
)
def test_gemini_reasoning_config_omit_include_thoughts(mock_chat, tmpdir):
    """Test Gemini without include_thoughts in reasoning_config."""
    credentials_json = generate_service_account_json_string()
    p = tmpdir.mkdir("sub").join("service-account.json")
    p.write(credentials_json)

    config_dict = {
        "name": "some_provider",
        "type": "google_vertex",
        "credentials_path": p.strpath,
        "project_id": "my-gcp-project",
        "models": [
            {
                "name": "gemini-2.5-flash",
                "parameters": {
                    "reasoning_config": {
                        "thinking_level": "medium",
                    }
                },
            }
        ],
    }
    provider_config = ProviderConfig(config_dict)
    vertex = GoogleVertex(
        model="gemini-2.5-flash", params={}, provider_config=provider_config
    )
    llm = vertex.load()
    assert llm is not None
    # When not specified in reasoning_config, include_thoughts should not be set
    assert "include_thoughts" not in vertex.default_params
    assert vertex.default_params["thinking_level"] == "medium"

    call_kwargs = mock_chat.call_args[1]
    assert "include_thoughts" not in call_kwargs
    assert call_kwargs["thinking_level"] == "medium"


@patch(
    "ols.src.llms.providers.google_vertex.ChatAnthropicVertex",
    autospec=True,
)
def test_anthropic_reasoning_config_with_thinking_enabled_high_effort(
    mock_chat, tmpdir
):
    """Test Anthropic on Vertex with thinking_enabled and high thinking_effort maps to adaptive."""
    credentials_json = generate_service_account_json_string()
    p = tmpdir.mkdir("sub").join("service-account.json")
    p.write(credentials_json)

    config_dict = {
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
                "parameters": {
                    "reasoning_config": {
                        "thinking_enabled": True,
                        "thinking_effort": "high",
                    }
                },
            }
        ],
    }
    provider_config = ProviderConfig(config_dict)
    vertex = GoogleVertexAnthropic(
        model="claude-opus-4-6", params={}, provider_config=provider_config
    )
    llm = vertex.load()
    assert llm is not None

    # Check that model_kwargs thinking config has adaptive type with summarized display
    model_kwargs = vertex.default_params.get("model_kwargs", {})
    thinking_config = model_kwargs.get("thinking", {})
    assert thinking_config["type"] == "adaptive"
    assert thinking_config["display"] == "summarized"

    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["model_kwargs"]["thinking"]["type"] == "adaptive"
    assert call_kwargs["model_kwargs"]["thinking"]["display"] == "summarized"


@patch(
    "ols.src.llms.providers.google_vertex.ChatAnthropicVertex",
    autospec=True,
)
def test_anthropic_reasoning_config_with_thinking_enabled_low_effort(mock_chat, tmpdir):
    """Test Anthropic on Vertex with thinking_enabled and low thinking_effort maps to enabled."""
    credentials_json = generate_service_account_json_string()
    p = tmpdir.mkdir("sub").join("service-account.json")
    p.write(credentials_json)

    config_dict = {
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
                "parameters": {
                    "reasoning_config": {
                        "thinking_enabled": True,
                        "thinking_effort": "low",
                    }
                },
            }
        ],
    }
    provider_config = ProviderConfig(config_dict)
    vertex = GoogleVertexAnthropic(
        model="claude-opus-4-6", params={}, provider_config=provider_config
    )
    llm = vertex.load()
    assert llm is not None

    # Check that model_kwargs thinking config has enabled type (not adaptive)
    model_kwargs = vertex.default_params.get("model_kwargs", {})
    thinking_config = model_kwargs.get("thinking", {})
    assert thinking_config["type"] == "enabled"
    assert "display" not in thinking_config

    call_kwargs = mock_chat.call_args[1]
    assert call_kwargs["model_kwargs"]["thinking"]["type"] == "enabled"
    assert "display" not in call_kwargs["model_kwargs"]["thinking"]


@patch(
    "ols.src.llms.providers.google_vertex.ChatAnthropicVertex",
    autospec=True,
)
def test_anthropic_reasoning_config_without_thinking_enabled(mock_chat, tmpdir):
    """Test Anthropic on Vertex without thinking_enabled does not set thinking config."""
    credentials_json = generate_service_account_json_string()
    p = tmpdir.mkdir("sub").join("service-account.json")
    p.write(credentials_json)

    config_dict = {
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
    provider_config = ProviderConfig(config_dict)
    vertex = GoogleVertexAnthropic(
        model="claude-opus-4-6", params={}, provider_config=provider_config
    )
    llm = vertex.load()
    assert llm is not None

    # Check that model_kwargs is not set when thinking_enabled is not provided
    model_kwargs = vertex.default_params.get("model_kwargs", {})
    assert "thinking" not in model_kwargs or model_kwargs.get("thinking", {}) == {}

    call_kwargs = mock_chat.call_args[1]
    assert (
        "model_kwargs" not in call_kwargs
        or call_kwargs.get("model_kwargs", {}).get("thinking", {}) == {}
    )
