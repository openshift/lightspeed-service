"""Unit tests for HITL endpoint handlers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from ols import config
from ols.app.models.config import HITLConfig

# needs to be setup there before imports
config.ols_config.authentication_config.module = "k8s"

from ols.app.endpoints import hitl  # noqa: E402
from ols.app.models.models import ApprovalRequest  # noqa: E402
from ols.src.hitl.hitl_manager import HITLManager, PendingApprovalState  # noqa: E402


@pytest.fixture
def hitl_config():
    """Create a HITL config for testing."""
    return HITLConfig(
        enabled=True,
        approval_timeout=300,
        tools_requiring_approval=[],
        auto_approve_read_only=True,
    )


@pytest.fixture
def hitl_manager(hitl_config):
    """Create a HITL manager for testing."""
    return HITLManager(hitl_config)


@pytest.fixture
def disabled_hitl_config():
    """Create a disabled HITL config for testing."""
    return HITLConfig(enabled=False)


@pytest.fixture
def disabled_hitl_manager(disabled_hitl_config):
    """Create a disabled HITL manager for testing."""
    return HITLManager(disabled_hitl_config)


class TestSubmitApprovalEndpoint:
    """Tests for the /approve endpoint."""

    @pytest.mark.asyncio
    async def test_submit_approval_hitl_disabled(self, disabled_hitl_manager):
        """Test that endpoint returns 400 when HITL is disabled."""
        with patch.object(
            config, "_hitl_manager", disabled_hitl_manager
        ):
            request = ApprovalRequest(
                approval_id="test-123",
                conversation_id="conv-456",
                decision="approve",
            )

            with pytest.raises(HTTPException) as exc_info:
                await hitl.submit_approval(request)

            assert exc_info.value.status_code == 400
            assert "not enabled" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_submit_approval_success(self, hitl_manager):
        """Test successful approval submission."""
        # Create a pending approval first
        approval_id, _ = await hitl_manager.request_approval(
            conversation_id="conv-456",
            tool_name="test_tool",
            tool_args={},
            tool_id="tool-789",
        )

        with patch.object(config, "_hitl_manager", hitl_manager):
            request = ApprovalRequest(
                approval_id=approval_id,
                conversation_id="conv-456",
                decision="approve",
            )

            response = await hitl.submit_approval(request)

            assert response.status == "accepted"
            assert response.approval_id == approval_id

    @pytest.mark.asyncio
    async def test_submit_approval_not_found(self, hitl_manager):
        """Test submitting approval for non-existent ID."""
        with patch.object(config, "_hitl_manager", hitl_manager):
            request = ApprovalRequest(
                approval_id="non-existent-id",
                conversation_id="conv-456",
                decision="approve",
            )

            with pytest.raises(HTTPException) as exc_info:
                await hitl.submit_approval(request)

            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_submit_approval_with_modification(self, hitl_manager):
        """Test approval with modified arguments."""
        approval_id, _ = await hitl_manager.request_approval(
            conversation_id="conv-456",
            tool_name="test_tool",
            tool_args={"namespace": "default"},
            tool_id="tool-789",
        )

        with patch.object(config, "_hitl_manager", hitl_manager):
            request = ApprovalRequest(
                approval_id=approval_id,
                conversation_id="conv-456",
                decision="modify",
                modified_args={"namespace": "production"},
            )

            response = await hitl.submit_approval(request)

            assert response.status == "accepted"


class TestGetPendingApprovalsEndpoint:
    """Tests for the /pending/{conversation_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_pending_approvals_hitl_disabled(self, disabled_hitl_manager):
        """Test that endpoint returns empty list when HITL is disabled."""
        with patch.object(config, "_hitl_manager", disabled_hitl_manager):
            response = await hitl.get_pending_approvals("conv-123")

            assert response.pending_approvals == []

    @pytest.mark.asyncio
    async def test_get_pending_approvals_empty(self, hitl_manager):
        """Test getting pending approvals when none exist."""
        with patch.object(config, "_hitl_manager", hitl_manager):
            response = await hitl.get_pending_approvals("conv-123")

            assert response.pending_approvals == []

    @pytest.mark.asyncio
    async def test_get_pending_approvals_with_results(self, hitl_manager):
        """Test getting pending approvals with existing approvals."""
        # Create some pending approvals
        await hitl_manager.request_approval(
            conversation_id="conv-123",
            tool_name="tool1",
            tool_args={"arg": "value"},
            tool_id="tool-1",
        )
        await hitl_manager.request_approval(
            conversation_id="conv-123",
            tool_name="tool2",
            tool_args={},
            tool_id="tool-2",
        )

        with patch.object(config, "_hitl_manager", hitl_manager):
            response = await hitl.get_pending_approvals("conv-123")

            assert len(response.pending_approvals) == 2
            tool_names = {p.tool_name for p in response.pending_approvals}
            assert tool_names == {"tool1", "tool2"}

    @pytest.mark.asyncio
    async def test_get_pending_approvals_filters_by_conversation(self, hitl_manager):
        """Test that results are filtered by conversation ID."""
        # Create approvals for different conversations
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

        with patch.object(config, "_hitl_manager", hitl_manager):
            response = await hitl.get_pending_approvals("conv-123")

            assert len(response.pending_approvals) == 1
            assert response.pending_approvals[0].tool_name == "tool1"
