"""Human-in-the-loop manager for tool approval workflows."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from ols.app.models.config import HITLConfig
from ols.utils import suid

logger = logging.getLogger(__name__)

READ_ONLY_TOOL_PREFIXES = ("get", "list", "describe", "read", "show", "fetch", "find")


@dataclass
class ApprovalDecision:
    """Represents a decision made on a tool approval request.

    Attributes:
        decision: The decision type (approve, reject, timeout).
        modified_args: Optional modified arguments if decision allows modification.
        approval_id: The ID of the approval request.
    """

    decision: Literal["approve", "reject", "timeout"]
    modified_args: Optional[dict[str, Any]] = None
    approval_id: str = ""

    @property
    def approved(self) -> bool:
        """Check if the decision is an approval."""
        return self.decision == "approve"

    @property
    def rejected(self) -> bool:
        """Check if the decision is a rejection or timeout."""
        return self.decision in ("reject", "timeout")


@dataclass
class PendingApprovalState:
    """Internal state for a pending approval request.

    Attributes:
        approval_id: Unique identifier for this approval.
        conversation_id: The conversation this approval belongs to.
        tool_name: Name of the tool awaiting approval.
        tool_args: Arguments for the tool call.
        tool_id: The tool call ID from the LLM.
        created_at: Timestamp when approval was requested.
        expires_at: Timestamp when approval will expire.
        event: Asyncio event to signal when decision is made.
        decision: The decision once made.
    """

    approval_id: str
    conversation_id: str
    tool_name: str
    tool_args: dict[str, Any]
    tool_id: str
    created_at: float
    expires_at: float
    event: asyncio.Event = field(default_factory=asyncio.Event)
    decision: Optional[ApprovalDecision] = None


class HITLManager:
    """Manager for human-in-the-loop tool approval workflows.

    This class handles the synchronization between streaming responses
    that pause for approval and the separate approval endpoint.
    """

    def __init__(self, config: HITLConfig) -> None:
        """Initialize the HITL manager.

        Args:
            config: HITL configuration settings.
        """
        self._config = config
        self._pending_approvals: dict[str, PendingApprovalState] = {}
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        """Check if HITL is enabled."""
        return self._config.enabled

    def requires_approval(self, tool_name: str, tool_args: dict[str, Any]) -> bool:
        """Check if a tool call requires human approval.

        Args:
            tool_name: Name of the tool being called.
            tool_args: Arguments for the tool call.

        Returns:
            True if approval is required, False otherwise.
        """
        if not self._config.enabled:
            return False

        if self._config.auto_approve_read_only:
            tool_lower = tool_name.lower()
            if any(tool_lower.startswith(prefix) for prefix in READ_ONLY_TOOL_PREFIXES):
                logger.debug(
                    "Auto-approving read-only tool: %s", tool_name
                )
                return False

        if self._config.tools_requiring_approval:
            return tool_name in self._config.tools_requiring_approval

        return True

    async def request_approval(
        self,
        conversation_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_id: str,
    ) -> tuple[str, PendingApprovalState]:
        """Request approval for a tool call.

        Creates a pending approval state and returns immediately.
        The caller should yield an approval_required event and then
        call wait_for_approval().

        Args:
            conversation_id: The conversation ID.
            tool_name: Name of the tool.
            tool_args: Arguments for the tool.
            tool_id: The tool call ID from the LLM.

        Returns:
            Tuple of (approval_id, pending_state).
        """
        approval_id = suid.get_suid()
        now = time.time()
        expires_at = now + self._config.approval_timeout

        state = PendingApprovalState(
            approval_id=approval_id,
            conversation_id=conversation_id,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_id=tool_id,
            created_at=now,
            expires_at=expires_at,
        )

        async with self._lock:
            self._pending_approvals[approval_id] = state

        logger.info(
            "Approval requested: id=%s, conversation=%s, tool=%s",
            approval_id,
            conversation_id,
            tool_name,
        )

        return approval_id, state

    async def wait_for_approval(
        self, approval_id: str, wait_timeout: Optional[float] = None
    ) -> ApprovalDecision:
        """Wait for an approval decision.

        Blocks until the approval is submitted or times out.

        Args:
            approval_id: The approval ID to wait for.
            wait_timeout: Optional timeout override (uses config default if None).

        Returns:
            The approval decision.
        """
        async with self._lock:
            state = self._pending_approvals.get(approval_id)

        if state is None:
            logger.error("Approval not found: %s", approval_id)
            return ApprovalDecision(decision="reject", approval_id=approval_id)

        effective_timeout = (
            wait_timeout if wait_timeout is not None else self._config.approval_timeout
        )

        try:
            await asyncio.wait_for(state.event.wait(), timeout=effective_timeout)

            if state.decision is not None:
                logger.info(
                    "Approval decision received: id=%s, decision=%s",
                    approval_id,
                    state.decision.decision,
                )
                return state.decision

            logger.warning("Approval event set but no decision: %s", approval_id)
            return ApprovalDecision(decision="reject", approval_id=approval_id)

        except asyncio.TimeoutError:
            logger.warning("Approval timeout: %s", approval_id)
            default_decision = self._config.default_on_timeout
            decision = ApprovalDecision(
                decision=default_decision if default_decision == "approve" else "timeout",
                approval_id=approval_id,
            )
            await self._cleanup_approval(approval_id)
            return decision

    async def submit_approval(
        self,
        approval_id: str,
        decision: Literal["approve", "reject", "modify"],
        modified_args: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, str]:
        """Submit an approval decision.

        Args:
            approval_id: The approval ID.
            decision: The decision (approve, reject, or modify).
            modified_args: Optional modified arguments for 'modify' decision.

        Returns:
            Tuple of (success, message).
        """
        async with self._lock:
            state = self._pending_approvals.get(approval_id)

        if state is None:
            logger.warning("Approval not found for submission: %s", approval_id)
            return False, "Approval not found or expired"

        if time.time() > state.expires_at:
            logger.warning("Approval expired: %s", approval_id)
            await self._cleanup_approval(approval_id)
            return False, "Approval has expired"

        effective_decision: Literal["approve", "reject", "timeout"] = (
            "approve" if decision in ("approve", "modify") else "reject"
        )

        state.decision = ApprovalDecision(
            decision=effective_decision,
            modified_args=modified_args if decision == "modify" else None,
            approval_id=approval_id,
        )
        state.event.set()

        logger.info(
            "Approval submitted: id=%s, decision=%s",
            approval_id,
            decision,
        )

        return True, f"Approval {decision}d successfully"

    async def get_pending_approvals(
        self, conversation_id: str
    ) -> list[PendingApprovalState]:
        """Get all pending approvals for a conversation.

        Args:
            conversation_id: The conversation ID.

        Returns:
            List of pending approval states.
        """
        now = time.time()
        async with self._lock:
            expired = [
                aid
                for aid, state in self._pending_approvals.items()
                if state.expires_at < now
            ]
            for aid in expired:
                del self._pending_approvals[aid]

            return [
                state
                for state in self._pending_approvals.values()
                if state.conversation_id == conversation_id
            ]

    async def _cleanup_approval(self, approval_id: str) -> None:
        """Remove an approval from pending state.

        Args:
            approval_id: The approval ID to remove.
        """
        async with self._lock:
            self._pending_approvals.pop(approval_id, None)

    async def cleanup_expired(self) -> int:
        """Remove all expired approvals.

        Returns:
            Number of expired approvals removed.
        """
        now = time.time()
        async with self._lock:
            expired = [
                aid
                for aid, state in self._pending_approvals.items()
                if state.expires_at < now
            ]
            for aid in expired:
                del self._pending_approvals[aid]
            return len(expired)

