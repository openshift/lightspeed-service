"""Handlers for tool approval decision endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from ols import config
from ols.app.models.models import (
    ErrorResponse,
    ForbiddenResponse,
    ToolApprovalDecisionRequest,
    UnauthorizedResponse,
)
from ols.src.auth.auth import get_auth_dependency
from ols.src.tools.approval import ApprovalSetResult, set_approval_decision

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tool-approvals", tags=["tool-approvals"])
auth_dependency = get_auth_dependency(config.ols_config, virtual_path="/ols-access")

submit_approval_responses: dict[int | str, dict[str, Any]] = {
    200: {"description": "Approval decision applied successfully"},
    401: {
        "description": "Missing or invalid credentials provided by client",
        "model": UnauthorizedResponse,
    },
    403: {
        "description": "Client does not have permission to access resource",
        "model": ForbiddenResponse,
    },
    404: {
        "description": "Approval request not found",
        "model": ErrorResponse,
    },
    409: {
        "description": "Approval request already resolved",
        "model": ErrorResponse,
    },
}


@router.post(
    "/decision",
    status_code=status.HTTP_200_OK,
    responses=submit_approval_responses,
)
async def submit_tool_approval_decision(
    request: ToolApprovalDecisionRequest,
    auth: tuple[str, str, bool, str] = Depends(auth_dependency),
) -> None:
    """Submit user decision for a pending tool approval request."""
    del auth  # Auth dependency enforces request authentication.

    result = set_approval_decision(request.approval_id, request.approved)
    match result:
        case ApprovalSetResult.APPLIED:
            return
        case ApprovalSetResult.NOT_FOUND:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "response": "Approval request not found",
                    "cause": f"No pending approval for approval_id {request.approval_id}",
                },
            )
        case ApprovalSetResult.ALREADY_RESOLVED:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "response": "Approval request already resolved",
                    "cause": f"Approval {request.approval_id} was already resolved",
                },
            )
