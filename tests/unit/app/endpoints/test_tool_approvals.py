"""Unit tests for tool approval decision endpoint handlers."""

from unittest.mock import patch

import pytest
from fastapi import HTTPException

from ols import config
from ols.src.tools.approval import ApprovalSetResult

# needs to be setup before tool_approvals endpoint is imported
config.ols_config.authentication_config.module = "k8s"

from ols.app.endpoints import tool_approvals  # noqa: E402
from ols.app.models.models import ToolApprovalDecisionRequest  # noqa: E402


@pytest.fixture
def mock_auth():
    """Create a mock auth tuple."""
    return ("test-user-id", "test-username", False, "test-token")


@pytest.mark.asyncio
async def test_submit_tool_approval_decision_applied(mock_auth):
    """Return success when approval decision is applied."""
    request = ToolApprovalDecisionRequest(approval_id="approval-1", approved=True)

    with patch(
        "ols.app.endpoints.tool_approvals.set_approval_decision",
        return_value=ApprovalSetResult.APPLIED,
    ) as mock_set_decision:
        response = await tool_approvals.submit_tool_approval_decision(
            request, auth=mock_auth
        )

    assert response is None
    mock_set_decision.assert_called_once_with("approval-1", True)


@pytest.mark.asyncio
async def test_submit_tool_approval_decision_not_found(mock_auth):
    """Return 404 when approval request is not found."""
    request = ToolApprovalDecisionRequest(approval_id="missing-approval", approved=True)

    with patch(
        "ols.app.endpoints.tool_approvals.set_approval_decision",
        return_value=ApprovalSetResult.NOT_FOUND,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await tool_approvals.submit_tool_approval_decision(request, auth=mock_auth)

    assert exc_info.value.status_code == 404
    assert "Approval request not found" in exc_info.value.detail["response"]


@pytest.mark.asyncio
async def test_submit_tool_approval_decision_already_resolved(mock_auth):
    """Return 409 when approval request was already resolved."""
    request = ToolApprovalDecisionRequest(
        approval_id="resolved-approval", approved=False
    )

    with patch(
        "ols.app.endpoints.tool_approvals.set_approval_decision",
        return_value=ApprovalSetResult.ALREADY_RESOLVED,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await tool_approvals.submit_tool_approval_decision(request, auth=mock_auth)

    assert exc_info.value.status_code == 409
    assert "Approval request already resolved" in exc_info.value.detail["response"]
