"""Unit tests for BAM provider."""

import pytest
from langchain_community.llms import FakeListLLM

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


def test_basic_interface(provider_config):
    """Test basic interface."""
    fake = FakeProvider(model="fake_model", params={}, provider_config=provider_config)
    llm = fake.load()
    assert isinstance(llm, FakeListLLM)
    assert fake.default_params is not None
