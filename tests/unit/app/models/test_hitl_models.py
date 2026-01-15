"""Unit tests for HITL models."""

import pytest
from pydantic import ValidationError

from ols.app.models.models import (
    ApprovalRequest,
    ApprovalResponse,
    PendingApproval,
    PendingApprovalsResponse,
)


class TestApprovalRequest:
    """Tests for ApprovalRequest model."""

    def test_valid_approval_request(self):
        """Test creating a valid approval request."""
        request = ApprovalRequest(
            approval_id="test-123",
            conversation_id="conv-456",
            decision="approve",
        )

        assert request.approval_id == "test-123"
        assert request.conversation_id == "conv-456"
        assert request.decision == "approve"
        assert request.modified_args is None

    def test_approval_request_with_modified_args(self):
        """Test approval request with modified arguments."""
        request = ApprovalRequest(
            approval_id="test-123",
            conversation_id="conv-456",
            decision="modify",
            modified_args={"namespace": "production"},
        )

        assert request.modified_args == {"namespace": "production"}

    def test_invalid_decision(self):
        """Test that invalid decision raises validation error."""
        with pytest.raises(ValidationError):
            ApprovalRequest(
                approval_id="test-123",
                conversation_id="conv-456",
                decision="invalid",
            )

    def test_all_valid_decisions(self):
        """Test all valid decision values."""
        for decision in ["approve", "reject", "modify"]:
            request = ApprovalRequest(
                approval_id="test-123",
                conversation_id="conv-456",
                decision=decision,
            )
            assert request.decision == decision


class TestApprovalResponse:
    """Tests for ApprovalResponse model."""

    def test_valid_approval_response(self):
        """Test creating a valid approval response."""
        response = ApprovalResponse(
            approval_id="test-123",
            status="accepted",
            message="Approval processed",
        )

        assert response.approval_id == "test-123"
        assert response.status == "accepted"
        assert response.message == "Approval processed"

    def test_response_without_message(self):
        """Test response with optional message omitted."""
        response = ApprovalResponse(
            approval_id="test-123",
            status="expired",
        )

        assert response.message is None

    def test_invalid_status(self):
        """Test that invalid status raises validation error."""
        with pytest.raises(ValidationError):
            ApprovalResponse(
                approval_id="test-123",
                status="invalid_status",
            )

    def test_all_valid_statuses(self):
        """Test all valid status values."""
        for status in ["accepted", "expired", "not_found", "error"]:
            response = ApprovalResponse(
                approval_id="test-123",
                status=status,
            )
            assert response.status == status


class TestPendingApproval:
    """Tests for PendingApproval model."""

    def test_valid_pending_approval(self):
        """Test creating a valid pending approval."""
        approval = PendingApproval(
            approval_id="test-123",
            conversation_id="conv-456",
            tool_name="kubectl_apply",
            tool_args={"yaml": "test"},
            created_at=1704657600.0,
            expires_at=1704657900.0,
        )

        assert approval.approval_id == "test-123"
        assert approval.conversation_id == "conv-456"
        assert approval.tool_name == "kubectl_apply"
        assert approval.tool_args == {"yaml": "test"}
        assert approval.created_at == 1704657600.0
        assert approval.expires_at == 1704657900.0

    def test_empty_tool_args(self):
        """Test pending approval with empty tool args."""
        approval = PendingApproval(
            approval_id="test-123",
            conversation_id="conv-456",
            tool_name="simple_tool",
            tool_args={},
            created_at=1704657600.0,
            expires_at=1704657900.0,
        )

        assert approval.tool_args == {}


class TestPendingApprovalsResponse:
    """Tests for PendingApprovalsResponse model."""

    def test_empty_pending_approvals(self):
        """Test response with no pending approvals."""
        response = PendingApprovalsResponse(pending_approvals=[])

        assert response.pending_approvals == []

    def test_multiple_pending_approvals(self):
        """Test response with multiple pending approvals."""
        approval1 = PendingApproval(
            approval_id="test-1",
            conversation_id="conv-123",
            tool_name="tool1",
            tool_args={},
            created_at=1704657600.0,
            expires_at=1704657900.0,
        )
        approval2 = PendingApproval(
            approval_id="test-2",
            conversation_id="conv-123",
            tool_name="tool2",
            tool_args={"key": "value"},
            created_at=1704657601.0,
            expires_at=1704657901.0,
        )

        response = PendingApprovalsResponse(pending_approvals=[approval1, approval2])

        assert len(response.pending_approvals) == 2
        assert response.pending_approvals[0].tool_name == "tool1"
        assert response.pending_approvals[1].tool_name == "tool2"



