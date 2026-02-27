"""Approval helper functions for tool execution."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal, TypeAlias

from ols.app.models.config import ApprovalType
from ols.utils.config import config

logger = logging.getLogger(__name__)

ApprovalOutcome: TypeAlias = Literal["approved", "rejected", "timeout", "error"]


@dataclass(slots=True)
class PendingApproval:
    """Store a single pending tool approval request."""

    approval_id: str
    decision: bool | None = None


class PendingApprovalStore:
    """In-memory store for pending tool approvals."""

    def __init__(self) -> None:
        """Initialize an empty pending approval store."""
        # approval_id -> pending approval state
        self._items: dict[str, PendingApproval] = {}

    def add(self, approval_id: str) -> PendingApproval:
        """Add or replace a pending approval request by approval_id."""
        # Create a fresh unresolved approval row for this approval request.
        pending = PendingApproval(approval_id=approval_id)
        # Upsert semantics are fine for in-memory flow; latest request wins.
        self._items[approval_id] = pending
        return pending

    def get(self, approval_id: str) -> PendingApproval | None:
        """Return pending approval by approval_id if present."""
        # Read-only lookup used by waiter and decision setter paths.
        return self._items.get(approval_id)

    def delete(self, approval_id: str) -> bool:
        """Delete pending approval by approval_id. Return False when not found."""
        # Remove completed/expired approval state from in-memory store.
        return self._items.pop(approval_id, None) is not None

    def set_decision(
        self, approval_id: str, approved: bool
    ) -> Literal["applied", "not_found", "already_resolved"]:
        """Persist approval decision for a pending request."""
        pending = self.get(approval_id)
        if pending is None:
            return "not_found"
        if pending.decision is not None:
            return "already_resolved"
        pending.decision = approved
        return "applied"


APPROVAL_POLL_INTERVAL_SECONDS = 0.5


def register_pending_approval(approval_id: str) -> None:
    """Register a pending approval request in storage.

    Args:
        approval_id: Unique approval request identifier to register.

    Returns:
        None.
    """
    config.pending_approval_store.add(approval_id)


async def get_approval_decision(
    approval_id: str,
    timeout_seconds: int,
) -> ApprovalOutcome:
    """Wait for decision and return approval result.

    Caller must register pending approval first via register_pending_approval().

    Args:
        approval_id: Unique approval request identifier to wait for.
        timeout_seconds: Maximum time to wait for decision before timing out.

    Returns:
        Approval decision outcome: approved, rejected, timeout, or error.
    """
    store = config.pending_approval_store

    # Keep wait logic local to this flow so storage remains CRUD-only.
    async def _wait_for_approval_decision() -> ApprovalOutcome:
        """Wait for approval decision using storage reads only.

        Args:
            None. Uses enclosing-scope values approval_id and timeout_seconds.

        Returns:
            Approval decision outcome.
        """
        # For in-memory state we could wait on an asyncio.Future, but we poll
        # intentionally so this waiter model stays compatible with a future
        # DB-backed approval store (same read-loop contract, different backend).
        loop = asyncio.get_running_loop()
        # Use monotonic loop time to avoid wall-clock jumps.
        deadline = loop.time() + float(timeout_seconds)
        while loop.time() < deadline:
            await asyncio.sleep(APPROVAL_POLL_INTERVAL_SECONDS)
            pending = store.get(approval_id)
            # Any persisted decision immediately completes the wait.
            if pending is not None and pending.decision is not None:
                if pending.decision:
                    return "approved"
                return "rejected"
        # Timed out while waiting for a client approval decision.
        logger.warning(
            "Approval decision timed out for approval_id=%s after %s seconds",
            approval_id,
            timeout_seconds,
        )
        return "timeout"

    try:
        # Suspend until decision/timeout while keeping cleanup in finally.
        return await _wait_for_approval_decision()
    except Exception:
        # Fail closed on unexpected waiting errors.
        logger.exception(
            "Unexpected error while waiting for approval_id=%s", approval_id
        )
        return "error"
    finally:
        # Always clean up in-memory state for this approval_id after completion.
        store.delete(approval_id)


def set_approval_decision(
    approval_id: str, approved: bool
) -> Literal["applied", "not_found", "already_resolved"]:
    """Set approval decision for a pending request.

    Args:
        approval_id: Unique approval request identifier to resolve.
        approved: Decision value where True means approved and False means rejected.

    Returns:
        "applied": decision was set successfully
        "not_found": no pending approval exists for approval_id
        "already_resolved": approval exists but was already completed
    """
    return config.pending_approval_store.set_decision(approval_id, approved)


def normalize_tool_annotation(
    tool_annotation: dict[str, object] | None,
) -> dict[str, object]:
    """Normalize tool metadata/annotation payload to annotation-only dict."""
    if not tool_annotation:
        return {}
    nested_annotation = tool_annotation.get("annotations")
    if isinstance(nested_annotation, dict):
        return nested_annotation
    return tool_annotation


def is_approval_enabled(
    streaming: bool,
    approval_type: ApprovalType | str,
) -> bool:
    """Return true when approval flow is enabled for the request."""
    # Current policy: approval workflow is supported only for streaming requests.
    if not streaming:
        return False
    # Normalize enum/string config value before strategy checks.
    approval_value = (
        approval_type.value
        if isinstance(approval_type, ApprovalType)
        else approval_type
    )
    # Approval flow is active only for explicit approval strategies.
    return approval_value in {
        ApprovalType.ALWAYS.value,
        ApprovalType.TOOL_ANNOTATIONS.value,
    }


def need_validation(
    streaming: bool,
    approval_type: ApprovalType | str,
    tool_annotation: dict[str, object] | None = None,
) -> bool:
    """Return true when a tool call must go through approval validation."""
    # Fast exit when approval flow is disabled for this request.
    if not is_approval_enabled(streaming=streaming, approval_type=approval_type):
        return False

    # Normalize enum/string config value before per-strategy decision.
    approval_value = (
        approval_type.value
        if isinstance(approval_type, ApprovalType)
        else approval_type
    )
    match approval_value:
        case ApprovalType.TOOL_ANNOTATIONS.value:
            # Annotation strategy: require approval by default unless the tool
            # explicitly declares readOnlyHint=true.
            annotation_payload = normalize_tool_annotation(tool_annotation)
            if not annotation_payload:
                return True
            value = annotation_payload.get("readOnlyHint")
            return not (isinstance(value, bool) and value)
        case _:
            # ALWAYS strategy (and any unknown fallback) requires approval.
            return True
