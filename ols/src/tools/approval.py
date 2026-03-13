"""Approval helper functions for tool execution."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

from ols.app.models.config import ApprovalType
from ols.utils.config import config

logger = logging.getLogger(__name__)


class ApprovalOutcome(StrEnum):
    """Outcome values for approval decision waiting."""

    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    ERROR = "error"


class ApprovalSetResult(StrEnum):
    """Result values for setting an approval decision."""

    APPLIED = "applied"
    NOT_FOUND = "not_found"
    ALREADY_RESOLVED = "already_resolved"


@dataclass(slots=True)
class PendingApproval:
    """Store a single pending tool approval request."""

    approval_id: str
    decision: bool | None = None
    event: asyncio.Event = field(default_factory=asyncio.Event)


class PendingApprovalStoreBase(ABC):
    """Abstract store contract for pending tool approvals."""

    @abstractmethod
    def add(self, approval_id: str) -> PendingApproval:
        """Add or replace a pending approval request by approval_id."""

    @abstractmethod
    def get(self, approval_id: str) -> PendingApproval | None:
        """Return pending approval by approval_id if present."""

    @abstractmethod
    def delete(self, approval_id: str) -> bool:
        """Delete pending approval by approval_id. Return False when not found."""

    @abstractmethod
    def set_decision(self, approval_id: str, approved: bool) -> ApprovalSetResult:
        """Persist approval decision for a pending request."""


class InMemoryPendingApprovalStore(PendingApprovalStoreBase):
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

    def set_decision(self, approval_id: str, approved: bool) -> ApprovalSetResult:
        """Persist approval decision for a pending request."""
        pending = self.get(approval_id)
        if pending is None:
            return ApprovalSetResult.NOT_FOUND
        if pending.decision is not None:
            return ApprovalSetResult.ALREADY_RESOLVED
        pending.decision = approved
        pending.event.set()
        return ApprovalSetResult.APPLIED


def create_pending_approval_store() -> PendingApprovalStoreBase:
    """Create the default pending approval store implementation."""
    return InMemoryPendingApprovalStore()


def register_pending_approval(approval_id: str) -> None:
    """Register a pending approval request in storage.

    Args:
        approval_id: Unique approval request identifier to register.
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

    pending = store.get(approval_id)
    if pending is None:
        logger.error("Pending approval not found for approval_id=%s", approval_id)
        return ApprovalOutcome.ERROR

    try:
        await asyncio.wait_for(pending.event.wait(), timeout=timeout_seconds)
        if pending.decision is True:
            return ApprovalOutcome.APPROVED
        if pending.decision is False:
            return ApprovalOutcome.REJECTED
        logger.error(
            "Approval event set but decision missing for approval_id=%s", approval_id
        )
        return ApprovalOutcome.ERROR
    except TimeoutError:
        logger.warning(
            "Approval decision timed out for approval_id=%s after %s seconds",
            approval_id,
            timeout_seconds,
        )
        return ApprovalOutcome.TIMEOUT
    except Exception:
        # Fail closed on unexpected waiting errors.
        logger.exception(
            "Unexpected error while waiting for approval_id=%s", approval_id
        )
        return ApprovalOutcome.ERROR
    finally:
        # Always clean up in-memory state for this approval_id after completion.
        store.delete(approval_id)


def set_approval_decision(approval_id: str, approved: bool) -> ApprovalSetResult:
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


def _approval_type_value(approval_type: ApprovalType | str) -> str:
    """Return string value for approval strategy enum/string input."""
    if isinstance(approval_type, ApprovalType):
        return approval_type.value
    return approval_type


def is_approval_enabled(
    streaming: bool,
    approval_type: ApprovalType | str,
) -> bool:
    """Return true when approval flow is enabled for the request."""
    # Current policy: approval workflow is supported only for streaming requests.
    if not streaming:
        return False
    # Normalize enum/string config value before strategy checks.
    approval_value = _approval_type_value(approval_type)
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
    approval_value = _approval_type_value(approval_type)
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
