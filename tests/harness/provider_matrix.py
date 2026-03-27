"""Cross-provider parametrization utilities.

Provides helpers that let a single test function run against multiple LLM
providers (OpenAI, Anthropic, Gemini) via an OpenAI-compatible proxy.

Usage::

    from tests.harness.provider_matrix import for_each_provider

    @for_each_provider
    @pytest.mark.asyncio
    async def test_tool_selection(provider_config):
        \"\"\"Verify the LLM selects the right tool.\"\"\"
        ...

The proxy URL and API key come from environment variables.  If neither is
set the entire harness is skipped gracefully — CI stays green.
"""

import os
from dataclasses import dataclass, field

import pytest

# ---------------------------------------------------------------------------
# Environment knobs
# ---------------------------------------------------------------------------
ENV_BASE_URL = "OLS_TEST_LLM_BASE_URL"
ENV_API_KEY = "OLS_TEST_LLM_API_KEY"
ENV_PROVIDER = "OLS_TEST_PROVIDER"

SUPPORTED_PROVIDERS = ("openai", "anthropic", "gemini")


# ---------------------------------------------------------------------------
# Provider configuration dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProviderConfig:
    """Immutable configuration for a single provider test run.

    Attributes:
        name: Provider identifier (``openai``, ``anthropic``, ``gemini``).
        base_url: OpenAI-compatible proxy base URL.
        api_key: API key for the proxy.
        model: Model identifier to use for this provider.
        extra: Provider-specific overrides (e.g. temperature, top_p).
    """

    name: str
    base_url: str
    api_key: str
    model: str = ""
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Default model mapping — override via OLS_TEST_MODEL_<PROVIDER>
# ---------------------------------------------------------------------------
_DEFAULT_MODELS: dict[str, str] = {
    "openai": os.environ.get("OLS_TEST_MODEL_OPENAI", "gpt-4o"),
    "anthropic": os.environ.get("OLS_TEST_MODEL_ANTHROPIC", "claude-sonnet-4-20250514"),
    "gemini": os.environ.get("OLS_TEST_MODEL_GEMINI", "gemini-2.5-pro"),
}


def _get_base_url() -> str | None:
    return os.environ.get(ENV_BASE_URL)


def _get_api_key() -> str | None:
    return os.environ.get(ENV_API_KEY)


# ---------------------------------------------------------------------------
# Build ProviderConfig from environment
# ---------------------------------------------------------------------------
def provider_config_for(name: str) -> ProviderConfig:
    """Build a ``ProviderConfig`` for *name* from environment variables.

    Raises:
        pytest.skip: When required env vars are missing.
    """
    base_url = _get_base_url()
    api_key = _get_api_key()
    if not base_url or not api_key:
        pytest.skip(
            f"{ENV_BASE_URL} and {ENV_API_KEY} must be set to run harness tests"
        )

    return ProviderConfig(
        name=name,
        base_url=base_url,
        api_key=api_key,
        model=_DEFAULT_MODELS.get(name, ""),
    )


# ---------------------------------------------------------------------------
# Parametrize decorator
# ---------------------------------------------------------------------------
def _requested_providers() -> list[str]:
    """Return the provider list requested via ``--provider`` or env var.

    Falls back to all supported providers.
    """
    env = os.environ.get(ENV_PROVIDER, "")
    if env.lower() == "all" or not env:
        return list(SUPPORTED_PROVIDERS)
    names = [p.strip().lower() for p in env.split(",")]
    unknown = set(names) - set(SUPPORTED_PROVIDERS)
    if unknown:
        raise ValueError(
            f"Unknown provider(s): {unknown}. "
            f"Supported: {SUPPORTED_PROVIDERS}"
        )
    return names


for_each_provider = pytest.mark.parametrize(
    "provider_name",
    _requested_providers(),
    ids=lambda name: f"provider={name}",
)
"""Decorator that parametrizes a test across requested LLM providers.

The test function receives a ``provider_name`` string parameter.  Use
``provider_config_for(provider_name)`` inside the test to get the full
``ProviderConfig``.
"""
