"""Unit tests for the providers module."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeChatModel

from ols import config, constants
from ols.app.models.config import ProviderConfig, TLSSecurityProfile
from ols.src.llms.providers.provider import LLMProvider, _no_proxy_mount_key
from ols.src.llms.providers.registry import (
    LLMProvidersRegistry,
    register_llm_provider_as,
)


def test_providers_are_registered():
    """Test providers are auto registered."""
    assert constants.PROVIDER_OPENAI in LLMProvidersRegistry.llm_providers
    assert constants.PROVIDER_WATSONX in LLMProvidersRegistry.llm_providers
    assert constants.PROVIDER_FAKE in LLMProvidersRegistry.llm_providers
    assert constants.PROVIDER_RHOAI_VLLM in LLMProvidersRegistry.llm_providers
    assert constants.PROVIDER_RHELAI_VLLM in LLMProvidersRegistry.llm_providers
    assert (
        constants.PROVIDER_GOOGLE_VERTEX_ANTHROPIC in LLMProvidersRegistry.llm_providers
    )
    assert constants.PROVIDER_GOOGLE_VERTEX in LLMProvidersRegistry.llm_providers
    assert constants.PROVIDER_BEDROCK in LLMProvidersRegistry.llm_providers

    # import after previous test to not influence the auto-registration
    from ols.src.llms.providers.bedrock import Bedrock
    from ols.src.llms.providers.fake_provider import FakeProvider
    from ols.src.llms.providers.google_vertex import GoogleVertex, GoogleVertexAnthropic
    from ols.src.llms.providers.openai import OpenAI
    from ols.src.llms.providers.rhelai_vllm import RHELAIVLLM
    from ols.src.llms.providers.rhoai_vllm import RHOAIVLLM
    from ols.src.llms.providers.watsonx import Watsonx

    assert LLMProvidersRegistry.llm_providers[constants.PROVIDER_OPENAI] == OpenAI
    assert LLMProvidersRegistry.llm_providers[constants.PROVIDER_WATSONX] == Watsonx
    assert (
        LLMProvidersRegistry.llm_providers[constants.PROVIDER_RHELAI_VLLM] == RHELAIVLLM
    )
    assert (
        LLMProvidersRegistry.llm_providers[constants.PROVIDER_RHOAI_VLLM] == RHOAIVLLM
    )
    assert LLMProvidersRegistry.llm_providers[constants.PROVIDER_FAKE] == FakeProvider
    assert (
        LLMProvidersRegistry.llm_providers[constants.PROVIDER_GOOGLE_VERTEX_ANTHROPIC]
        == GoogleVertexAnthropic
    )
    assert (
        LLMProvidersRegistry.llm_providers[constants.PROVIDER_GOOGLE_VERTEX]
        == GoogleVertex
    )
    assert LLMProvidersRegistry.llm_providers[constants.PROVIDER_BEDROCK] == Bedrock


def test_valid_provider_is_registered():
    """Test valid (`LLMProvider` subclass) is registered."""

    @register_llm_provider_as("spam")
    class Spam(LLMProvider):
        @property
        def default_params(self):
            return {}

        def load(self):
            return FakeChatModel()

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
            return FakeChatModel()

    my_provider = MyProvider(
        model="bla", params={"provider-param": 2}, provider_config=None
    )

    assert my_provider.params["provider-param"] == 2
    assert my_provider.params["not-to-be-overwritten-param"] == "foo"


def test_llm_provider_params_order__config_overrides_everything():
    """Test config params overrides llm params."""
    config.dev_config.llm_params = {"provider-param": 3}

    class MyProvider(LLMProvider):
        @property
        def default_params(self):
            return {"provider-param": 1, "not-to-be-overwritten-param": "foo"}

        def load(self):
            return FakeChatModel()

    my_provider = MyProvider(
        model="bla", params={"provider-param": 2}, provider_config=None
    )

    assert my_provider.params["provider-param"] == 3
    assert my_provider.params["not-to-be-overwritten-param"] == "foo"


def test_llm_provider_params_order__no_provider_type():
    """Test how missing provider type is handled."""
    config.dev_config.llm_params = {"provider-param": 3}

    class MyProvider(LLMProvider):
        @property
        def default_params(self):
            return {"provider-param": 1, "not-to-be-overwritten-param": "foo"}

        def load(self):
            return FakeChatModel()

    # set up provider configuration with type set to None
    provider_config = ProviderConfig()
    provider_config.type = None

    my_provider = MyProvider(model="bla", params={}, provider_config=provider_config)

    assert my_provider.params["provider-param"] == 3
    assert my_provider.params["not-to-be-overwritten-param"] == "foo"


def test_construct_httpx_client():
    """Test the HTTPX client construction."""

    class MyProvider(LLMProvider):
        @property
        def default_params(self):
            return {"provider-param": 1, "not-to-be-overwritten-param": "foo"}

        def load(self):
            return FakeChatModel()

    # set up provider configuration with type set to None
    provider_config = ProviderConfig()
    provider_config.type = None
    provider_config.tls_security_profile = TLSSecurityProfile(
        {
            "type": "Custom",
            "minTLSVersion": "VersionTLS12",
            "ciphers": None,
        }
    )
    llm_provider = MyProvider("model", provider_config)
    client = llm_provider._construct_httpx_client(False)
    assert client is not None


# --- Tests for _no_proxy_mount_key ---


def test_no_proxy_mount_key_bare_ipv6_loopback():
    """Bare IPv6 loopback is bracketed and uses exact-match (no wildcard)."""
    assert _no_proxy_mount_key("::1") == "all://[::1]"


def test_no_proxy_mount_key_bare_ipv6_full():
    """A full bare IPv6 address is bracketed and uses exact-match."""
    assert _no_proxy_mount_key("2001:db8::1") == "all://[2001:db8::1]"


def test_no_proxy_mount_key_already_bracketed_ipv6():
    """An already-bracketed IPv6 address is not double-bracketed."""
    assert _no_proxy_mount_key("[::1]") == "all://[::1]"


def test_no_proxy_mount_key_hostname():
    """Plain hostnames use the wildcard suffix pattern."""
    assert _no_proxy_mount_key("example.com") == "all://*example.com"


def test_no_proxy_mount_key_ipv4():
    """IPv4 addresses use the wildcard suffix pattern."""
    assert _no_proxy_mount_key("192.168.1.1") == "all://*192.168.1.1"


def test_no_proxy_mount_key_cidr():
    """CIDR ranges use the wildcard suffix pattern."""
    assert _no_proxy_mount_key("10.0.0.0/8") == "all://*10.0.0.0/8"


def test_no_proxy_mount_key_host_with_port():
    """host:port entries (single colon) are treated as plain hostnames, not IPv6."""
    assert _no_proxy_mount_key("internal-api:8443") == "all://*internal-api:8443"


def test_construct_httpx_client_with_ipv6_in_no_proxy_hosts():
    """httpx client construction must not crash when ::1 is in no_proxy_hosts.

    Regression test for OLS-3736: bare IPv6 in the cluster's no-proxy list
    caused httpx to raise 'Invalid port: :1' when the mounts dict was built
    with the unbracketed address embedded directly in a URL pattern.
    """

    class MyProvider(LLMProvider):
        @property
        def default_params(self):
            return {}

        def load(self):
            return FakeChatModel()

    provider_config = ProviderConfig()
    provider_config.type = None
    provider_config.tls_security_profile = None
    llm_provider = MyProvider("model", provider_config)

    mock_proxy_config = MagicMock()
    # proxy_url=None → the proxy construction block is skipped
    mock_proxy_config.proxy_url = None
    mock_proxy_config.no_proxy_hosts = ["::1", "[::1]", "localhost", "10.0.0.0/8"]

    with patch("ols.src.llms.providers.provider.config") as mock_config:
        mock_config.ols_config.proxy_config = mock_proxy_config
        mock_config.dev_config.llm_params = None
        client = llm_provider._construct_httpx_client(False, False)
        assert client is not None
