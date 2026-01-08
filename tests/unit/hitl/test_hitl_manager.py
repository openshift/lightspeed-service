"""Unit tests for HITL manager."""

import asyncio
from unittest.mock import MagicMock

import pytest

from ols.app.models.config import HITLConfig
from ols.src.hitl.hitl_manager import ApprovalDecision, HITLManager


@pytest.fixture
def hitl_config():
    """Create a default HITL config for testing."""
    return HITLConfig(
        enabled=True,
        approval_timeout=5,
        tools_requiring_approval=[],
        auto_approve_read_only=True,
        default_on_timeout="reject",
    )


@pytest.fixture
def disabled_hitl_config():
    """Create a disabled HITL config for testing."""
    return HITLConfig(enabled=False)


@pytest.fixture
def hitl_manager(hitl_config):
    """Create a HITL manager instance."""
    return HITLManager(hitl_config)


@pytest.fixture
def disabled_hitl_manager(disabled_hitl_config):
    """Create a disabled HITL manager instance."""
    return HITLManager(disabled_hitl_config)


class TestHITLManagerEnabled:
    """Tests for enabled HITL manager."""

    def test_enabled_property(self, hitl_manager):
        """Test that enabled property returns True."""
        assert hitl_manager.enabled is True

    def test_disabled_property(self, disabled_hitl_manager):
        """Test that enabled property returns False when disabled."""
        assert disabled_hitl_manager.enabled is False


class TestRequiresApproval:
    """Tests for requires_approval method."""

    def test_requires_approval_when_disabled(self, disabled_hitl_manager):
        """Test that approval is not required when HITL is disabled."""
        assert disabled_hitl_manager.requires_approval("any_tool", {}) is False

    def test_requires_approval_for_regular_tool(self, hitl_manager):
        """Test that approval is required for non-read-only tools."""
        assert hitl_manager.requires_approval("kubectl_apply", {}) is True
        assert hitl_manager.requires_approval("create_pod", {}) is True
        assert hitl_manager.requires_approval("delete_namespace", {}) is True

    def test_auto_approve_read_only_tools(self, hitl_manager):
        """Test that read-only tools are auto-approved."""
        assert hitl_manager.requires_approval("get_pods", {}) is False
        assert hitl_manager.requires_approval("list_namespaces", {}) is False
        assert hitl_manager.requires_approval("describe_deployment", {}) is False
        assert hitl_manager.requires_approval("read_configmap", {}) is False
        assert hitl_manager.requires_approval("show_logs", {}) is False
        assert hitl_manager.requires_approval("fetch_events", {}) is False
        assert hitl_manager.requires_approval("find_resources", {}) is False

    def test_specific_tools_requiring_approval(self, hitl_config):
        """Test that only specified tools require approval."""
        hitl_config.tools_requiring_approval = ["dangerous_tool", "another_tool"]
        manager = HITLManager(hitl_config)

        assert manager.requires_approval("dangerous_tool", {}) is True
        assert manager.requires_approval("another_tool", {}) is True
        assert manager.requires_approval("safe_tool", {}) is False

    def test_read_only_override_disabled(self, hitl_config):
        """Test that read-only auto-approve can be disabled."""
        hitl_config.auto_approve_read_only = False
        manager = HITLManager(hitl_config)

        assert manager.requires_approval("get_pods", {}) is True
        assert manager.requires_approval("list_namespaces", {}) is True


class TestApprovalDecision:
    """Tests for ApprovalDecision dataclass."""

    def test_approved_property(self):
        """Test approved property."""
        decision = ApprovalDecision(decision="approve")
        assert decision.approved is True
        assert decision.rejected is False

    def test_rejected_property(self):
        """Test rejected property."""
        decision = ApprovalDecision(decision="reject")
        assert decision.approved is False
        assert decision.rejected is True

    def test_timeout_is_rejected(self):
        """Test that timeout is considered rejected."""
        decision = ApprovalDecision(decision="timeout")
        assert decision.approved is False
        assert decision.rejected is True

    def test_modified_args(self):
        """Test modified_args storage."""
        decision = ApprovalDecision(
            decision="approve",
            modified_args={"namespace": "production"},
            approval_id="test-123",
        )
        assert decision.modified_args == {"namespace": "production"}
        assert decision.approval_id == "test-123"


class TestRequestApproval:
    """Tests for request_approval method."""

    @pytest.mark.asyncio
    async def test_request_approval_creates_state(self, hitl_manager):
        """Test that request_approval creates a pending approval state."""
        approval_id, state = await hitl_manager.request_approval(
            conversation_id="conv-123",
            tool_name="kubectl_apply",
            tool_args={"yaml": "test"},
            tool_id="tool-456",
        )

        assert approval_id is not None
        assert state.conversation_id == "conv-123"
        assert state.tool_name == "kubectl_apply"
        assert state.tool_args == {"yaml": "test"}
        assert state.tool_id == "tool-456"
        assert state.decision is None
        assert not state.event.is_set()

    @pytest.mark.asyncio
    async def test_request_approval_sets_expiry(self, hitl_manager):
        """Test that request_approval sets correct expiry time."""
        approval_id, state = await hitl_manager.request_approval(
            conversation_id="conv-123",
            tool_name="test_tool",
            tool_args={},
            tool_id="tool-456",
        )

        # Expiry should be ~5 seconds in the future (from config)
        assert state.expires_at > state.created_at
        assert state.expires_at - state.created_at == pytest.approx(5.0, abs=0.1)


class TestSubmitApproval:
    """Tests for submit_approval method."""

    @pytest.mark.asyncio
    async def test_submit_approval_success(self, hitl_manager):
        """Test successful approval submission."""
        approval_id, state = await hitl_manager.request_approval(
            conversation_id="conv-123",
            tool_name="test_tool",
            tool_args={},
            tool_id="tool-456",
        )

        success, message = await hitl_manager.submit_approval(
            approval_id=approval_id,
            decision="approve",
        )

        assert success is True
        assert "successfully" in message.lower()
        assert state.event.is_set()
        assert state.decision.approved is True

    @pytest.mark.asyncio
    async def test_submit_rejection(self, hitl_manager):
        """Test rejection submission."""
        approval_id, state = await hitl_manager.request_approval(
            conversation_id="conv-123",
            tool_name="test_tool",
            tool_args={},
            tool_id="tool-456",
        )

        success, message = await hitl_manager.submit_approval(
            approval_id=approval_id,
            decision="reject",
        )

        assert success is True
        assert state.decision.rejected is True

    @pytest.mark.asyncio
    async def test_submit_with_modified_args(self, hitl_manager):
        """Test approval with modified arguments."""
        approval_id, state = await hitl_manager.request_approval(
            conversation_id="conv-123",
            tool_name="test_tool",
            tool_args={"namespace": "default"},
            tool_id="tool-456",
        )

        success, _ = await hitl_manager.submit_approval(
            approval_id=approval_id,
            decision="modify",
            modified_args={"namespace": "production"},
        )

        assert success is True
        assert state.decision.approved is True
        assert state.decision.modified_args == {"namespace": "production"}

    @pytest.mark.asyncio
    async def test_submit_approval_not_found(self, hitl_manager):
        """Test submitting approval for non-existent ID."""
        success, message = await hitl_manager.submit_approval(
            approval_id="non-existent-id",
            decision="approve",
        )

        assert success is False
        assert "not found" in message.lower()


class TestWaitForApproval:
    """Tests for wait_for_approval method."""

    @pytest.mark.asyncio
    async def test_wait_for_approval_receives_decision(self, hitl_manager):
        """Test that wait_for_approval returns when approval is submitted."""
        approval_id, state = await hitl_manager.request_approval(
            conversation_id="conv-123",
            tool_name="test_tool",
            tool_args={},
            tool_id="tool-456",
        )

        # Submit approval in background
        async def submit_later():
            await asyncio.sleep(0.1)
            await hitl_manager.submit_approval(approval_id, "approve")

        asyncio.create_task(submit_later())

        decision = await hitl_manager.wait_for_approval(approval_id)

        assert decision.approved is True

    @pytest.mark.asyncio
    async def test_wait_for_approval_timeout(self, hitl_config):
        """Test that wait_for_approval times out correctly."""
        hitl_config.approval_timeout = 0.5
        manager = HITLManager(hitl_config)

        approval_id, _ = await manager.request_approval(
            conversation_id="conv-123",
            tool_name="test_tool",
            tool_args={},
            tool_id="tool-456",
        )

        decision = await manager.wait_for_approval(approval_id)

        assert decision.decision == "timeout"
        assert decision.rejected is True

    @pytest.mark.asyncio
    async def test_wait_for_approval_timeout_auto_approve(self, hitl_config):
        """Test auto-approve on timeout."""
        hitl_config.approval_timeout = 0.5
        hitl_config.default_on_timeout = "approve"
        manager = HITLManager(hitl_config)

        approval_id, _ = await manager.request_approval(
            conversation_id="conv-123",
            tool_name="test_tool",
            tool_args={},
            tool_id="tool-456",
        )

        decision = await manager.wait_for_approval(approval_id)

        assert decision.decision == "approve"
        assert decision.approved is True

    @pytest.mark.asyncio
    async def test_wait_for_nonexistent_approval(self, hitl_manager):
        """Test waiting for non-existent approval."""
        decision = await hitl_manager.wait_for_approval("non-existent-id")

        assert decision.rejected is True


class TestGetPendingApprovals:
    """Tests for get_pending_approvals method."""

    @pytest.mark.asyncio
    async def test_get_pending_approvals_empty(self, hitl_manager):
        """Test getting pending approvals when none exist."""
        pending = await hitl_manager.get_pending_approvals("conv-123")
        assert pending == []

    @pytest.mark.asyncio
    async def test_get_pending_approvals_filters_by_conversation(self, hitl_manager):
        """Test that pending approvals are filtered by conversation ID."""
        await hitl_manager.request_approval(
            conversation_id="conv-123",
            tool_name="tool1",
            tool_args={},
            tool_id="tool-1",
        )
        await hitl_manager.request_approval(
            conversation_id="conv-456",
            tool_name="tool2",
            tool_args={},
            tool_id="tool-2",
        )
        await hitl_manager.request_approval(
            conversation_id="conv-123",
            tool_name="tool3",
            tool_args={},
            tool_id="tool-3",
        )

        pending = await hitl_manager.get_pending_approvals("conv-123")

        assert len(pending) == 2
        tool_names = {s.tool_name for s in pending}
        assert tool_names == {"tool1", "tool3"}

    @pytest.mark.asyncio
    async def test_get_pending_approvals_excludes_expired(self, hitl_config):
        """Test that expired approvals are excluded."""
        hitl_config.approval_timeout = 0.1
        manager = HITLManager(hitl_config)

        await manager.request_approval(
            conversation_id="conv-123",
            tool_name="tool1",
            tool_args={},
            tool_id="tool-1",
        )

        # Wait for expiry
        await asyncio.sleep(0.2)

        pending = await manager.get_pending_approvals("conv-123")
        assert pending == []


class TestCleanupExpired:
    """Tests for cleanup_expired method."""

    @pytest.mark.asyncio
    async def test_cleanup_expired_removes_old_approvals(self, hitl_config):
        """Test that cleanup_expired removes expired approvals."""
        hitl_config.approval_timeout = 0.1
        manager = HITLManager(hitl_config)

        await manager.request_approval(
            conversation_id="conv-123",
            tool_name="tool1",
            tool_args={},
            tool_id="tool-1",
        )
        await manager.request_approval(
            conversation_id="conv-123",
            tool_name="tool2",
            tool_args={},
            tool_id="tool-2",
        )

        # Wait for expiry
        await asyncio.sleep(0.2)

        removed = await manager.cleanup_expired()

        assert removed == 2



