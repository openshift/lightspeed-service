"""Integration tests for tool approval decision endpoint."""

# we add new attributes into pytest instance, which is not recognized
# properly by linters
# pyright: reportAttributeAccessIssue=false

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ols import config
from ols.src.tools.approval import ApprovalSetResult


@pytest.fixture(scope="function")
def _setup() -> None:
    """Set up TestClient with integration configuration."""
    config.reload_from_yaml_file("tests/config/config_for_integration_tests.yaml")

    # app.main must be imported after config is loaded
    from ols.app.main import app  # pylint: disable=import-outside-toplevel

    pytest.client = TestClient(app)


def test_submit_tool_approval_decision_applied(_setup: None) -> None:
    """Return HTTP 200 when decision is applied."""
    with patch(
        "ols.app.endpoints.tool_approvals.set_approval_decision",
        return_value=ApprovalSetResult.APPLIED,
    ) as mock_set_decision:
        response = pytest.client.post(  # type: ignore[attr-defined]
            "/v1/tool-approvals/decision",
            json={"approval_id": "approval-1", "approved": True},
        )

    assert response.status_code == 200
    assert response.json() is None
    mock_set_decision.assert_called_once_with("approval-1", True)


def test_submit_tool_approval_decision_not_found(_setup: None) -> None:
    """Return HTTP 404 when approval request is missing."""
    with patch(
        "ols.app.endpoints.tool_approvals.set_approval_decision",
        return_value=ApprovalSetResult.NOT_FOUND,
    ):
        response = pytest.client.post(  # type: ignore[attr-defined]
            "/v1/tool-approvals/decision",
            json={"approval_id": "missing-approval", "approved": True},
        )

    assert response.status_code == 404
    assert response.json()["detail"]["response"] == "Approval request not found"


def test_submit_tool_approval_decision_already_resolved(_setup: None) -> None:
    """Return HTTP 409 when approval request was already resolved."""
    with patch(
        "ols.app.endpoints.tool_approvals.set_approval_decision",
        return_value=ApprovalSetResult.ALREADY_RESOLVED,
    ):
        response = pytest.client.post(  # type: ignore[attr-defined]
            "/v1/tool-approvals/decision",
            json={"approval_id": "resolved-approval", "approved": False},
        )

    assert response.status_code == 409
    assert response.json()["detail"]["response"] == "Approval request already resolved"
