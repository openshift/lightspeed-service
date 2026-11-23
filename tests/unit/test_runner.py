"""Tests for runner startup dependencies."""

from types import SimpleNamespace

import pytest

import runner
from ols.src.rag_index import rhokp_mcp


def test_load_index_waits_for_rhokp_before_loading_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initialize RHOKP MCP before exposing the service through Uvicorn."""
    events: list[str] = []
    settings = object()

    class ConfigStub:
        ols_config = SimpleNamespace(openshift_docs_mcp=settings)

        @property
        def rag_index(self) -> None:
            events.append("rag_index")
            return None

    async def initialize_rhokp_mcp(settings: object) -> None:
        assert settings is ConfigStub.ols_config.openshift_docs_mcp
        events.append("rhokp_mcp")

    monkeypatch.setattr(runner, "config", ConfigStub(), raising=False)
    monkeypatch.setattr(rhokp_mcp, "initialize_rhokp_mcp", initialize_rhokp_mcp)

    runner.load_index()

    assert events == ["rhokp_mcp", "rag_index"]
