"""Unit tests for approval module."""

import asyncio

import pytest

from ols.app.models.config import ApprovalType
from ols.src.tools import approval as approval_module
from ols.src.tools.approval import (
    PendingApprovalStore,
    get_approval_decision,
    is_approval_enabled,
    need_validation,
    normalize_tool_annotation,
    register_pending_approval,
    set_approval_decision,
)


def test_pending_approval_store_crud() -> None:
    """Test add/get/delete lifecycle for in-memory approval store."""

    async def _run() -> None:
        store = PendingApprovalStore()
        approval_id = "approval-1"

        pending = store.add(approval_id)
        assert pending.approval_id == approval_id
        assert store.get(approval_id) is pending
        assert store.delete(approval_id) is True
        assert store.get(approval_id) is None
        assert store.delete(approval_id) is False

    asyncio.run(_run())


def test_get_approval_decision_returns_true_when_approved() -> None:
    """Test approval decision waiter returns True when approved."""

    async def _run() -> None:
        store = PendingApprovalStore()
        approval_module.config._pending_approval_store = store
        approval_id = "approval-2"
        register_pending_approval(approval_id)

        wait_task = asyncio.create_task(get_approval_decision(approval_id, 1))
        await asyncio.sleep(0)

        assert set_approval_decision(approval_id, True) == "applied"
        assert await wait_task == "approved"
        assert store.get(approval_id) is None

    asyncio.run(_run())


def test_get_approval_decision_returns_false_on_timeout() -> None:
    """Test approval decision waiter returns False when timeout occurs."""

    async def _run() -> None:
        store = PendingApprovalStore()
        approval_module.config._pending_approval_store = store
        approval_id = "approval-timeout"
        register_pending_approval(approval_id)

        result = await get_approval_decision(approval_id, timeout_seconds=0)
        assert result == "timeout"
        assert store.get(approval_id) is None

    asyncio.run(_run())


def test_get_approval_decision_returns_false_on_wait_error(
    monkeypatch,
) -> None:
    """Test approval decision waiter fails closed on unexpected wait errors."""

    async def _raise_runtime_error(*args: object, **kwargs: object) -> None:
        raise RuntimeError("wait failure")

    monkeypatch.setattr(approval_module.asyncio, "sleep", _raise_runtime_error)

    async def _run() -> None:
        store = PendingApprovalStore()
        approval_module.config._pending_approval_store = store
        approval_id = "approval-error"
        register_pending_approval(approval_id)

        result = await get_approval_decision(approval_id, timeout_seconds=1)
        assert result == "error"
        assert store.get(approval_id) is None

    asyncio.run(_run())


def test_get_approval_decision_keeps_waiting_when_row_temporarily_missing(
    monkeypatch,
) -> None:
    """Test waiter tolerates transient missing rows until timeout/decision."""
    monkeypatch.setattr(approval_module, "APPROVAL_POLL_INTERVAL_SECONDS", 0.01)

    async def _run() -> None:
        store = PendingApprovalStore()
        approval_module.config._pending_approval_store = store
        approval_id = "approval-flaky-row"
        register_pending_approval(approval_id)
        original_get = store.get
        calls = {"count": 0}

        def _flaky_get(target_approval_id: str):
            calls["count"] += 1
            if calls["count"] <= 2:
                return None
            return original_get(target_approval_id)

        monkeypatch.setattr(store, "get", _flaky_get)

        wait_task = asyncio.create_task(
            get_approval_decision(approval_id, timeout_seconds=1)
        )
        await asyncio.sleep(0.03)
        assert set_approval_decision(approval_id, True) == "applied"
        assert await wait_task == "approved"

    asyncio.run(_run())


def test_set_approval_decision_states() -> None:
    """Test set_approval_decision return states."""

    async def _run() -> None:
        store = PendingApprovalStore()
        approval_module.config._pending_approval_store = store

        assert set_approval_decision("missing", True) == "not_found"

        pending = store.add("approval-3")
        assert set_approval_decision("approval-3", False) == "applied"
        assert pending.decision is False
        assert set_approval_decision("approval-3", True) == "already_resolved"

    asyncio.run(_run())


def test_normalize_tool_annotation_variants() -> None:
    """Test annotation normalization for nested and direct payloads."""
    assert normalize_tool_annotation(None) == {}
    assert normalize_tool_annotation({}) == {}
    assert normalize_tool_annotation({"annotations": {"readOnlyHint": True}}) == {
        "readOnlyHint": True
    }
    assert normalize_tool_annotation({"readOnlyHint": False}) == {"readOnlyHint": False}


@pytest.mark.parametrize(
    ("streaming", "approval_type", "expected"),
    [
        (False, ApprovalType.ALWAYS, False),
        (True, ApprovalType.NEVER, False),
        (True, ApprovalType.ALWAYS, True),
        (True, ApprovalType.TOOL_ANNOTATIONS, True),
        (True, ApprovalType.ALWAYS.value, True),
    ],
)
def test_is_approval_enabled(
    streaming: bool, approval_type: ApprovalType | str, expected: bool
) -> None:
    """Test approval enablement policy."""
    assert is_approval_enabled(streaming, approval_type) is expected


def test_need_validation_policies() -> None:
    """Test per-tool validation policy combinations."""
    # Non-streaming always disables approval validation.
    assert need_validation(False, ApprovalType.ALWAYS, None) is False

    # Streaming + ALWAYS always validates.
    assert need_validation(True, ApprovalType.ALWAYS, None) is True

    # Streaming + TOOL_ANNOTATIONS defaults to validate when annotation missing.
    assert need_validation(True, ApprovalType.TOOL_ANNOTATIONS, None) is True
    assert need_validation(True, ApprovalType.TOOL_ANNOTATIONS, {}) is True

    # readOnlyHint=True skips validation.
    assert (
        need_validation(
            True,
            ApprovalType.TOOL_ANNOTATIONS,
            {"annotations": {"readOnlyHint": True}},
        )
        is False
    )
    assert (
        need_validation(
            True,
            ApprovalType.TOOL_ANNOTATIONS,
            {"readOnlyHint": True},
        )
        is False
    )

    # Non-boolean readOnlyHint does not disable validation.
    assert (
        need_validation(
            True,
            ApprovalType.TOOL_ANNOTATIONS,
            {"readOnlyHint": "true"},
        )
        is True
    )
