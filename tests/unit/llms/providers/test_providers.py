"""Unit tests for the providers module."""

import pytest

from ols import constants
from ols.src.llms.providers.provider import LLMProvider
from ols.src.llms.providers.registry import (
    LLMProvidersRegistry,
    register_llm_provider_as,
)
from ols.utils import config


def test_providers_are_registered():
    """Test providers are auto registered."""
    assert constants.PROVIDER_OPENAI in LLMProvidersRegistry.llm_providers
    assert constants.PROVIDER_BAM in LLMProvidersRegistry.llm_providers
    assert constants.PROVIDER_WATSONX in LLMProvidersRegistry.llm_providers

    # import after previous test to not influence the auto-registration
    from ols.src.llms.providers.bam import BAM
    from ols.src.llms.providers.openai import OpenAI
    from ols.src.llms.providers.watsonx import WatsonX

    assert LLMProvidersRegistry.llm_providers[constants.PROVIDER_OPENAI] == OpenAI
    assert LLMProvidersRegistry.llm_providers[constants.PROVIDER_BAM] == BAM
    assert LLMProvidersRegistry.llm_providers[constants.PROVIDER_WATSONX] == WatsonX


def test_valid_provider_is_registered():
    """Test valid (`LLMProvider` subclass) is registered."""

    @register_llm_provider_as("spam")
    class Spam(LLMProvider):
        @property
        def default_params(self):
            return

        def load(self):
            return

    assert "spam" in LLMProvidersRegistry.llm_providers


def test_invalid_provider_is_not_registered():
    """Test raise when invalid (not `LLMProvider` subclass) is registered."""
    with pytest.raises(ValueError, match="LLMProvider subclass required"):

        @register_llm_provider_as("spam")
        class Spam:
            pass


def test_llm_provider_params_order__inputs_overrides_defaults():
    """Test LLMProvider overrides default params."""
    config.init_empty_config()

    class MyProvider(LLMProvider):
        @property
        def default_params(self):
            return {"provider-param": 1}

        def load(self):
            return

    my_provider = MyProvider(
        model="bla", params={"provider-param": 2}, provider_config=None
    )

    assert my_provider.params["provider-param"] == 2


def test_llm_provider_params_order__config_overrides_everything():
    """Test config params overrides llm params."""
    config.init_empty_config()
    config.dev_config.llm_params = {"provider-param": 3}

    class MyProvider(LLMProvider):
        @property
        def default_params(self):
            return {"provider-param": 1}

        def load(self):
            return

    my_provider = MyProvider(
        model="bla", params={"provider-param": 2}, provider_config=None
    )

    assert my_provider.params["provider-param"] == 3
