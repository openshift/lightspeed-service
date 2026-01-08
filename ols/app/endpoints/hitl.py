"""Human-in-the-loop endpoints for tool approval workflows."""

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from ols import config
from ols.app.models.models import (
    ApprovalRequest,
    ApprovalResponse,
    ErrorResponse,
    ForbiddenResponse,
    PendingApproval,
    PendingApprovalsResponse,
    UnauthorizedResponse,
)
from ols.src.auth.auth import get_auth_dependency

logger = logging.getLogger(__name__)

router = APIRouter(tags=["hitl"])
auth_dependency = get_auth_dependency(config.ols_config, virtual_path="/ols-access")

approve_responses: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "Approval submitted successfully",
        "model": ApprovalResponse,
    },
    400: {
        "description": "Invalid approval request",
        "model": ErrorResponse,
    },
    401: {
        "description": "Missing or invalid credentials provided by client",
        "model": UnauthorizedResponse,
    },
    403: {
        "description": "Client does not have permission to access resource",
        "model": ForbiddenResponse,
    },
    404: {
        "description": "Approval not found or expired",
        "model": ErrorResponse,
    },
}


@router.post("/approve", responses=approve_responses)
async def submit_approval(
    request: ApprovalRequest,
    auth: Any = Depends(auth_dependency),
    user_id: Optional[str] = None,
) -> ApprovalResponse:
    """Submit an approval decision for a pending tool call.

    Args:
        request: The approval request containing the decision.
        auth: Authentication handler (FastAPI Depends).
        user_id: Optional user ID when no-op auth is enabled.

    Returns:
        ApprovalResponse indicating the result of the submission.
    """
    if not config.hitl_manager.enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "response": "HITL not enabled",
                "cause": "Human-in-the-loop feature is not enabled in configuration",
            },
        )

    success, message = await config.hitl_manager.submit_approval(
        approval_id=request.approval_id,
        decision=request.decision,
        modified_args=request.modified_args,
    )

    if not success:
        if "not found" in message.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "response": "Approval not found",
                    "cause": message,
                },
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "response": "Approval failed",
                "cause": message,
            },
        )

    logger.info(
        "Approval submitted: approval_id=%s, conversation_id=%s, decision=%s",
        request.approval_id,
        request.conversation_id,
        request.decision,
    )

    return ApprovalResponse(
        approval_id=request.approval_id,
        status="accepted",
        message=message,
    )


pending_responses: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "List of pending approvals",
        "model": PendingApprovalsResponse,
    },
    401: {
        "description": "Missing or invalid credentials provided by client",
        "model": UnauthorizedResponse,
    },
    403: {
        "description": "Client does not have permission to access resource",
        "model": ForbiddenResponse,
    },
}


@router.get("/pending/{conversation_id}", responses=pending_responses)
async def get_pending_approvals(
    conversation_id: str,
    auth: Any = Depends(auth_dependency),
    user_id: Optional[str] = None,
) -> PendingApprovalsResponse:
    """Get all pending approval requests for a conversation.

    Args:
        conversation_id: The conversation ID to get pending approvals for.
        auth: Authentication handler (FastAPI Depends).
        user_id: Optional user ID when no-op auth is enabled.

    Returns:
        PendingApprovalsResponse containing list of pending approvals.
    """
    if not config.hitl_manager.enabled:
        return PendingApprovalsResponse(pending_approvals=[])

    pending_states = await config.hitl_manager.get_pending_approvals(conversation_id)

    pending_approvals = [
        PendingApproval(
            approval_id=state.approval_id,
            conversation_id=state.conversation_id,
            tool_name=state.tool_name,
            tool_args=state.tool_args,
            created_at=state.created_at,
            expires_at=state.expires_at,
        )
        for state in pending_states
    ]

    return PendingApprovalsResponse(pending_approvals=pending_approvals)



