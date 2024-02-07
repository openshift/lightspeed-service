"""Unit tests for llm_loader module."""

from ols.src.llms.llm_loader import (
    LLMConfigurationError,
    ModelConfigMissingError,
    UnknownProviderError,
    UnsupportedProviderError,
)

# from unittest.mock import MagicMock, patch

# import pytest

# from ols.app.models.config import LLMProviders

# from ols.src.llms.providers.provider import LLMProvider
# from ols.src.llms.providers.registry import register_llm_provider_as
# from ols.utils import config


# @pytest.fixture
# def _registered_fake_provider():
#     """Register fake provider."""

#     @register_llm_provider_as("fake-provider")
#     class FakeProvider(LLMProvider):
#         @property
#         def default_params(self):
#             return {}

#         def load(self):
#             return "fake_llm"


def test_errors_relationship():
    """Test the relationship between LLMConfigurationError and its subclasses."""
    assert issubclass(UnknownProviderError, LLMConfigurationError)
    assert issubclass(UnsupportedProviderError, LLMConfigurationError)
    assert issubclass(ModelConfigMissingError, LLMConfigurationError)


# def test_constructor_unknown_provider():
#     """Test raises when provider is unknown."""
#     providers = LLMProviders()  # no providers
#     config.init_empty_config()
#     config.config.llm_providers = providers

#     msg = "Provider 'unknown-provider' is not a valid provider"
#     with pytest.raises(UnknownProviderError, match=msg):
#         LLMLoader(provider="unknown-provider", model="some model")


# def test_constructor_unsupported_provider():
#     """Test how a configured but an unsupported provider (not openai, bam, etc) is checked for."""
#     test_provider = "test-provider"
#     test_provider_type = "unsupported-provider-type"
#     config.config = Config(
#         {
#             "llm_providers": [
#                 {
#                     "name": test_provider,
#                     "type": "bam",
#                     "models": [
#                         {
#                             "name": constants.GRANITE_13B_CHAT_V1,
#                         }
#                     ],
#                 }
#             ]
#         }
#     )
#     config.config.llm_providers.providers[test_provider].type = test_provider_type
#     config.llm_config = config.config.llm_providers

#     with pytest.raises(UnsupportedProviderError):
#         LLMLoader(provider=test_provider, model=constants.GRANITE_13B_CHAT_V1)


# def test_constructor_unknown_model():
#     """Test raise when model configuration is missing."""
#     providers = LLMProviders([{"name": "bam", "models": [{"name": "model"}]}])
#     config.init_empty_config()
#     config.config.llm_providers = providers

#     message = "Model 'bla' is not a valid model for provider"
#     with pytest.raises(ModelConfigMissingError, match=message):
#         LLMLoader(provider="bam", model="bla")


# @patch("ols.src.llms.llm_loader.LLMProvidersRegistry", new=MagicMock())
# def test_unsupported_provider_error():
#     """Test raise when provider is not in the registry (not implemented)."""
#     providers = LLMProviders(
#         [{"name": "some-provider", "type": "bam", "models": [{"name": "model"}]}]
#     )
#     config.init_empty_config()
#     config.config.llm_providers = providers

#     with pytest.raises(UnsupportedProviderError):
#         load_llm(provider="some-provider", model="model")


# @patch("ols.constants.SUPPORTED_PROVIDER_TYPES", new=["fake-provider"])
# def test_load_llm(registered_fake_provider):
#     """Test load_llm function."""
#     providers = LLMProviders(
#         [
#             {
#                 "name": "fake-provider",
#                 "type": "fake-provider",
#                 "models": [{"name": "model"}],
#             }
#         ]
#     )
#     config.init_empty_config()
#     config.config.llm_providers = providers

#     llm = load_llm(provider="fake-provider", model="model")
#     assert llm == "fake_llm"
