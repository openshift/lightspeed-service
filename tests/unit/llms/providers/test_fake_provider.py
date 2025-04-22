"""Unit tests for BAM provider."""

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
                "response": "Hello",
                "chunks": 30,
                "sleep": 0.1,
            },
        }
    )


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
