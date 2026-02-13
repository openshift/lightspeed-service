"""Unit tests for BAM provider."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_community.llms import FakeListLLM
from langchain_community.llms.fake import FakeStreamingListLLM

from ols.app.models.config import ProviderConfig
from ols.src.llms.providers.fake_provider import FakeProvider


@pytest.fixture
def provider_config():
    """Fixture with provider configuration for FakeProvider."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "fake_provider",
            "models": [
                {
                    "name": "fake_model",
                }
            ],
        }
    )


@pytest.fixture
def provider_streaming_config():
    """Fixture with provider configuration for streaming enabled FakeProvider."""
    return ProviderConfig(
        {
            "name": "some_provider",
            "type": "fake_provider",
            "models": [
                {
                    "name": "fake_model",
                }
            ],
            "fake_provider_config": {
                "url": "http://example.com",  # some URL
                "stream": True,
                "mcp_tool_call": False,
                "response": "Hello",
                "chunks": 30,
                "sleep": 0.1,
            },
        }
    )


@patch("ols.src.llms.providers.fake_provider.requests.post")
def test_dynamic_response_success(mock_post, provider_streaming_config):
    """Test that MCP result is appended on success."""
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"items": ["deployment1"]}
    mock_post.return_value = mock_response
    provider_streaming_config.fake_provider_config.mcp_tool_call = True
    provider_streaming_config.fake_provider_config.stream = False

    fake = FakeProvider(
        model="fake_model", params={}, provider_config=provider_streaming_config
    )
    llm = fake.load()

    assert isinstance(llm, FakeListLLM)
    output = llm.responses[0]

    assert "MCP Result" in output
    assert "deployment1" in str(output)


@patch("ols.src.llms.providers.fake_provider.requests.post")
def test_dynamic_response_failure(mock_post, provider_streaming_config):
    """Test that base response is returned when MCP call fails."""
    mock_post.side_effect = Exception("Connection error")
    provider_streaming_config.fake_provider_config.mcp_tool_call = True
    provider_streaming_config.fake_provider_config.stream = False

    fake = FakeProvider(
        model="fake_model", params={}, provider_config=provider_streaming_config
    )
    llm = fake.load()

    output = llm.responses[0]

    # Should NOT include MCP result section
    assert "MCP Result" not in output
    assert fake.response in output


@patch("ols.src.llms.providers.fake_provider.requests.post")
def test_streaming_with_mcp(mock_post, provider_streaming_config):
    """Ensure streaming still respects chunk size with dynamic response."""
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"items": ["deployment1"]}
    mock_post.return_value = mock_response
    provider_streaming_config.fake_provider_config.mcp_tool_call = True

    fake = FakeProvider(
        model="fake_model",
        params={},
        provider_config=provider_streaming_config,
    )

    llm = fake.load()

    assert isinstance(llm, FakeStreamingListLLM)
    assert len(llm.responses[0]) == fake.default_params.get("chunks")


def test_bind_tools_exists(provider_config):
    """Ensure bind tools override exists."""
    fake = FakeProvider(model="fake_model", params={}, provider_config=provider_config)
    llm = fake.load()
    llm.bind_tools(tools=[])

    assert hasattr(llm, "bind_tools")


def test_basic_interface(provider_config):
    """Test basic interface."""
    fake = FakeProvider(model="fake_model", params={}, provider_config=provider_config)
    llm = fake.load()
    assert isinstance(llm, FakeListLLM)
    assert fake.default_params is not None


def test_streaming_interface(provider_streaming_config):
    """Test the interface for FakeStreamingListLLM."""
    fake = FakeProvider(
        model="fake_model", params={}, provider_config=provider_streaming_config
    )
    llm = fake.load()
    assert isinstance(llm, FakeStreamingListLLM)
    assert fake.default_params is not None
    assert fake.default_params.get("response") == "Hello"
    assert len(llm.responses[0]) == fake.default_params.get("chunks")
    assert llm.sleep == 0.1
