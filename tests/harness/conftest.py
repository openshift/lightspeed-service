"""Pytest configuration for the agent loop test harness.

Provides fixtures and CLI options for running integration-level tests
with fake OCP tools against a real LLM (via an OpenAI-compatible proxy)
or with a fully mocked LLM for offline unit testing.
"""

import os

import pytest

from ols import config

from tests.harness.fake_tools import ALL_FAKE_TOOLS, POLICY_MAP, SAFE_TOOLS
from tests.harness.provider_matrix import (
    ENV_API_KEY,
    ENV_BASE_URL,
    ProviderConfig,
    provider_config_for,
)


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    """Register harness-specific CLI options."""
    parser.addoption(
        "--provider",
        action="store",
        default="all",
        help="LLM provider to test: openai, anthropic, gemini, or all (default: all)",
    )


# ---------------------------------------------------------------------------
# Autouse: reset OLS config between tests (matches unit test pattern)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="function", autouse=True)
def _reset_ols_config() -> None:
    """Reset the OLS config singleton before each test."""
    config.reload_empty()


# ---------------------------------------------------------------------------
# Tool fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def fake_tools():
    """Return all six fake OCP tools as a ``list[StructuredTool]``."""
    return list(ALL_FAKE_TOOLS)


@pytest.fixture()
def safe_tools():
    """Return only the ALLOW-policy tools."""
    return list(SAFE_TOOLS)


@pytest.fixture()
def policy_map():
    """Return the tool-name → policy mapping dict."""
    return dict(POLICY_MAP)


# ---------------------------------------------------------------------------
# Provider fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def provider_config(request: pytest.FixtureRequest) -> ProviderConfig:
    """Build a ``ProviderConfig`` for the current parametrized provider.

    Expects ``provider_name`` to be injected via ``@for_each_provider``.
    Falls back to the ``--provider`` CLI option or the ``OLS_TEST_PROVIDER``
    env var.
    """
    name = getattr(request, "param", None)
    if name is None:
        name = request.config.getoption("--provider", default="openai")
    return provider_config_for(name)


@pytest.fixture()
def llm_proxy_url() -> str:
    """Return the LLM proxy base URL or skip if unset."""
    url = os.environ.get(ENV_BASE_URL)
    if not url:
        pytest.skip(f"{ENV_BASE_URL} not set — skipping live LLM tests")
    return url


@pytest.fixture()
def llm_api_key() -> str:
    """Return the LLM proxy API key or skip if unset."""
    key = os.environ.get(ENV_API_KEY)
    if not key:
        pytest.skip(f"{ENV_API_KEY} not set — skipping live LLM tests")
    return key
