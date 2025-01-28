"""Unit tests for the providers module."""

import pytest

from ols import config, constants
from ols.app.models.config import ProviderConfig
from ols.src.llms.providers.provider import LLMProvider
from ols.src.llms.providers.registry import (
    LLMProvidersRegistry,
    register_llm_provider_as,
)


def test_providers_are_registered():
    """Test providers are auto registered."""
    assert constants.PROVIDER_OPENAI in LLMProvidersRegistry.llm_providers
    assert constants.PROVIDER_BAM in LLMProvidersRegistry.llm_providers
    assert constants.PROVIDER_WATSONX in LLMProvidersRegistry.llm_providers
    assert constants.PROVIDER_FAKE in LLMProvidersRegistry.llm_providers
    assert constants.PROVIDER_RHOAI_VLLM in LLMProvidersRegistry.llm_providers
    assert constants.PROVIDER_RHELAI_VLLM in LLMProvidersRegistry.llm_providers

    # import after previous test to not influence the auto-registration
    from ols.src.llms.providers.bam import BAM
    from ols.src.llms.providers.fake_provider import FakeProvider
    from ols.src.llms.providers.openai import OpenAI
    from ols.src.llms.providers.rhelai_vllm import RHELAIVLLM
    from ols.src.llms.providers.rhoai_vllm import RHOAIVLLM
    from ols.src.llms.providers.watsonx import Watsonx

    assert LLMProvidersRegistry.llm_providers[constants.PROVIDER_OPENAI] == OpenAI
    assert LLMProvidersRegistry.llm_providers[constants.PROVIDER_BAM] == BAM
    assert LLMProvidersRegistry.llm_providers[constants.PROVIDER_WATSONX] == Watsonx
    assert (
        LLMProvidersRegistry.llm_providers[constants.PROVIDER_RHELAI_VLLM] == RHELAIVLLM
    )
    assert (
        LLMProvidersRegistry.llm_providers[constants.PROVIDER_RHOAI_VLLM] == RHOAIVLLM
    )
    assert LLMProvidersRegistry.llm_providers[constants.PROVIDER_FAKE] == FakeProvider


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
    with pytest.raises(TypeError, match="LLMProvider subclass required"):

        @register_llm_provider_as("spam")
        class Spam:
            pass


def test_llm_provider_params_order__inputs_overrides_defaults():
    """Test LLMProvider overrides default params."""

    class MyProvider(LLMProvider):
        @property
        def default_params(self):
            return {"provider-param": 1, "not-to-be-overwritten-param": "foo"}

        def load(self):
            return

    my_provider = MyProvider(
        model="bla", params={"provider-param": 2}, provider_config=None
    )

    assert my_provider.params["provider-param"] == 2
    assert my_provider.params["not-to-be-overwritten-param"] == "foo"
    assert my_provider.params["streaming"] is False

    my_provider_with_streaming_disabled = MyProvider(
        model="bla",
        params={"provider-param": 2},
        provider_config=None,
        streaming=False,
    )

    assert my_provider_with_streaming_disabled.params["provider-param"] == 2
    assert (
        my_provider_with_streaming_disabled.params["not-to-be-overwritten-param"]
        == "foo"
    )
    assert my_provider_with_streaming_disabled.params["streaming"] is False

    my_provider_with_streaming_enabled = MyProvider(
        model="bla", params={"provider-param": 2}, provider_config=None, streaming=True
    )

    assert my_provider_with_streaming_enabled.params["provider-param"] == 2
    assert (
        my_provider_with_streaming_enabled.params["not-to-be-overwritten-param"]
        == "foo"
    )
    assert my_provider_with_streaming_enabled.params["streaming"] is True


def test_llm_provider_params_order__config_overrides_everything():
    """Test config params overrides llm params."""
    config.dev_config.llm_params = {"provider-param": 3}

    class MyProvider(LLMProvider):
        @property
        def default_params(self):
            return {"provider-param": 1, "not-to-be-overwritten-param": "foo"}

        def load(self):
            return

    my_provider = MyProvider(
        model="bla", params={"provider-param": 2}, provider_config=None
    )

    assert my_provider.params["provider-param"] == 3
    assert my_provider.params["not-to-be-overwritten-param"] == "foo"

    my_provider_with_streaming_disabled = MyProvider(
        model="bla", params={"provider-param": 2}, provider_config=None, streaming=False
    )

    assert my_provider_with_streaming_disabled.params["provider-param"] == 3
    assert (
        my_provider_with_streaming_disabled.params["not-to-be-overwritten-param"]
        == "foo"
    )
    assert my_provider_with_streaming_disabled.params["streaming"] is False

    my_provider_with_streaming_enabled = MyProvider(
        model="bla", params={"provider-param": 2}, provider_config=None, streaming=True
    )

    assert my_provider_with_streaming_enabled.params["provider-param"] == 3
    assert (
        my_provider_with_streaming_enabled.params["not-to-be-overwritten-param"]
        == "foo"
    )
    assert my_provider_with_streaming_enabled.params["streaming"] is True


def test_llm_provider_params_order__no_provider_type():
    """Test how missing provider type is handled."""
    config.dev_config.llm_params = {"provider-param": 3}

    class MyProvider(LLMProvider):
        @property
        def default_params(self):
            return {"provider-param": 1, "not-to-be-overwritten-param": "foo"}

        def load(self):
            return

    # set up provider configuration with type set to None
    provider_config = ProviderConfig()
    provider_config.type = None

    my_provider = MyProvider(model="bla", params={}, provider_config=provider_config)

    assert my_provider.params["provider-param"] == 3
    assert my_provider.params["not-to-be-overwritten-param"] == "foo"

    my_provider_with_streaming_disabled = MyProvider(
        model="bla", params={"provider-param": 2}, provider_config=None, streaming=False
    )

    assert my_provider_with_streaming_disabled.params["provider-param"] == 3
    assert (
        my_provider_with_streaming_disabled.params["not-to-be-overwritten-param"]
        == "foo"
    )
    assert my_provider_with_streaming_disabled.params["streaming"] is False

    my_provider_with_streaming_enabled = MyProvider(
        model="bla", params={"provider-param": 2}, provider_config=None, streaming=True
    )

    assert my_provider_with_streaming_enabled.params["provider-param"] == 3
    assert (
        my_provider_with_streaming_enabled.params["not-to-be-overwritten-param"]
        == "foo"
    )
    assert my_provider_with_streaming_enabled.params["streaming"] is True
