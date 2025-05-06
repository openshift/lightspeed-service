"""Unit tests for llm_loader module."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeChatModel

from ols import config, constants
from ols.app.models.config import LLMProviders
from ols.src.llms.llm_loader import (
    LLMConfigurationError,
    ModelConfigMissingError,
    UnknownProviderError,
    UnsupportedProviderError,
    load_llm,
)
from ols.src.llms.providers.provider import LLMProvider
from ols.src.llms.providers.registry import register_llm_provider_as


@pytest.fixture
def _registered_fake_provider():
    """Register fake provider."""

    @register_llm_provider_as("fake-provider")
    class FakeProvider(LLMProvider):
        @property
        def default_params(self):
            return {}

        def load(self):
            return FakeChatModel()


def test_errors_relationship():
    """Test the relationship between LLMConfigurationError and its subclasses."""
    assert issubclass(UnknownProviderError, LLMConfigurationError)
    assert issubclass(UnsupportedProviderError, LLMConfigurationError)
    assert issubclass(ModelConfigMissingError, LLMConfigurationError)


def test_unknown_provider_error():
    """Test raise when provider is not in configuration."""
    providers = LLMProviders()  # no providers
    config.config.llm_providers = providers

    msg = "Provider 'unknown-provider' is not a valid provider"
    with pytest.raises(UnknownProviderError, match=msg):
        load_llm(provider="unknown-provider", model="some model")


def test_model_config_missing_error():
    """Test raise when model configuration is unknown."""
    providers = LLMProviders([{"name": "bam", "models": [{"name": "model"}]}])
    config.config.llm_providers = providers

    message = "Model 'bla' is not a valid model for provider"
    with pytest.raises(ModelConfigMissingError, match=message):
        load_llm(provider="bam", model="bla")


def test_unsupported_provider_error():
    """Test raise when provider is not in the registry (not implemented)."""
    providers = LLMProviders(
        [{"name": "some-provider", "type": "bam", "models": [{"name": "model"}]}]
    )
    config.config.llm_providers = providers

    with (
        patch("ols.src.llms.llm_loader.LLMProvidersRegistry", new=MagicMock()),
        pytest.raises(UnsupportedProviderError),
    ):
        load_llm(provider="some-provider", model="model")


@pytest.mark.usefixtures("_registered_fake_provider")
def test_load_llm():
    """Test load_llm function."""
    with patch("ols.constants.SUPPORTED_PROVIDER_TYPES", new=["fake-provider"]):
        providers = LLMProviders(
            [
                {
                    "name": "fake-provider",
                    "type": "fake-provider",
                    "models": [{"name": "model"}],
                }
            ]
        )
        config.config.llm_providers = providers

        llm = load_llm(provider="fake-provider", model="model")
        assert llm == FakeChatModel()


def test_load_llm_no_provider_config():
    """Test load_llm function."""
    config.config.llm_providers = None

    with pytest.raises(
        LLMConfigurationError,
        match=f"Providers configuration missing in {constants.DEFAULT_CONFIGURATION_FILE}",
    ):
        load_llm(provider="fake-provider", model="model")
