"""Unit tests for e2e marker collection by OLSConfig suffix."""

from types import SimpleNamespace

import pytest

from tests.e2e import conftest


class _Item:
    """Represent a pytest item with one optional marker."""

    def __init__(self, name: str, marker: str | None, config: object) -> None:
        self.name = name
        self._marker = marker
        self.config = config

    def get_closest_marker(self, name: str) -> object | None:
        """Return a marker token when the item has the requested marker."""
        return object() if self._marker == name else None


def test_cluster_session_gathers_before_mock_server_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep mock logs available to must-gather before removing the mock Pod."""
    events: list[str] = []
    monkeypatch.setattr(conftest, "on_cluster", True)
    monkeypatch.setenv("OLS_CONFIG_SUFFIX", "mcp_inspection")
    monkeypatch.setattr(conftest, "must_gather", lambda: events.append("gather"))
    monkeypatch.setattr(
        conftest, "teardown_mcp_on_cluster", lambda: events.append("teardown")
    )

    conftest.pytest_sessionfinish()

    assert events == ["gather", "teardown"]


def test_cluster_session_tears_down_mock_server_when_gather_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clean up test resources even when artifact collection fails."""
    events: list[str] = []
    monkeypatch.setattr(conftest, "on_cluster", True)
    monkeypatch.setenv("OLS_CONFIG_SUFFIX", "mcp_inspection")

    def fail_gather() -> None:
        events.append("gather")
        raise RuntimeError("artifact collection failed")

    monkeypatch.setattr(conftest, "must_gather", fail_gather)
    monkeypatch.setattr(
        conftest, "teardown_mcp_on_cluster", lambda: events.append("teardown")
    )

    with pytest.raises(RuntimeError, match="artifact collection failed"):
        conftest.pytest_sessionfinish()

    assert events == ["gather", "teardown"]


def test_inspection_marker_runs_only_with_dedicated_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deselect inspection tests outside the dedicated suffix."""
    deselected: list[_Item] = []
    hook = SimpleNamespace(pytest_deselected=lambda items: deselected.extend(items))
    config = SimpleNamespace(hook=hook)
    items = [
        _Item("inspection_test", "inspection", config),
        _Item("mcp_test", "mcp", config),
        _Item("regular_test", None, config),
    ]

    monkeypatch.setenv("OLS_CONFIG_SUFFIX", "default")
    conftest.pytest_collection_modifyitems(items)

    assert [item.name for item in items] == ["regular_test"]
    assert {item.name for item in deselected} == {"inspection_test", "mcp_test"}


def test_inspection_marker_runs_with_mcp_inspection_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep inspection and MCP tests selected for the dedicated suffix."""
    config = SimpleNamespace(hook=SimpleNamespace(pytest_deselected=lambda items: None))
    items = [
        _Item("inspection_test", "inspection", config),
        _Item("mcp_test", "mcp", config),
        _Item("regular_test", None, config),
    ]

    monkeypatch.setenv("OLS_CONFIG_SUFFIX", "mcp_inspection")
    conftest.pytest_collection_modifyitems(items)

    assert [item.name for item in items] == [
        "inspection_test",
        "mcp_test",
        "regular_test",
    ]
