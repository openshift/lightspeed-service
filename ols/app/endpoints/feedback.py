"""Handler for REST API call to provide user feedback."""

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ols.app.endpoints.ols import retrieve_user_id
from ols.app.models.models import (
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    ForbiddenResponse,
    StatusResponse,
    UnauthorizedResponse,
)
from ols.utils import config
from ols.utils.auth_dependency import AuthDependency
from ols.utils.suid import get_suid

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/feedback", tags=["feedback"])
auth_dependency = AuthDependency(virtual_path="/ols-access")


async def ensure_feedback_enabled(request: Request) -> None:
    """Check if feedback is enabled.

    Args:
        request (Request): The FastAPI request object.

    Raises:
        HTTPException: If feedback is disabled.
    """
    feedback_enabled = is_feedback_enabled()
    if not feedback_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Feedback is currently disabled.",
        )


def is_feedback_enabled() -> bool:
    """Check if feedback is enabled.

    Returns:
        True if feedback is enabled, False otherwise.
    """
    return not config.ols_config.user_data_collection.feedback_disabled


def store_feedback(user_id: str, feedback: dict) -> None:
    """Store feedback in the local filesystem.

    Args:
        user_id: The user ID (UUID).
        feedback: The feedback to store.
    """
    # ensures storage path exists
    storage_path = Path(config.ols_config.user_data_collection.feedback_storage)
    if not storage_path.exists():
        logger.debug(f"creating feedback storage directories '{storage_path}'")
        storage_path.mkdir(parents=True)

    data_to_store = {"user_id": user_id, **feedback}

    # stores feedback in a file under unique uuid
    feedback_file_path = storage_path / f"{get_suid()}.json"
    with open(feedback_file_path, "w", encoding="utf-8") as feedback_file:
        json.dump(data_to_store, feedback_file)

    logger.debug(f"feedback stored in '{feedback_file_path}'")


@router.get("/status")
def feedback_status() -> StatusResponse:
    """Handle feedback status requests.

    Returns:
        Response indicating the status of the feedback.
    """
    logger.debug("feedback status request received")
    feedback_status = is_feedback_enabled()
    return StatusResponse(functionality="feedback", status={"enabled": feedback_status})


post_feedback_responses = {
    200: {
        "description": "Feedback received and stored",
        "model": FeedbackResponse,
    },
    401: {
        "description": "Missing or invalid credentials provided by client",
        "model": UnauthorizedResponse,
    },
    403: {
        "description": "Client does not have permission to access resource",
        "model": ForbiddenResponse,
    },
    500: {
        "description": "User feedback can not be stored",
        "model": ErrorResponse,
    },
}


@router.post("", responses=post_feedback_responses)
def store_user_feedback(
    feedback_request: FeedbackRequest,
    ensure_feedback_enabled: Any = Depends(ensure_feedback_enabled),
    auth: Any = Depends(auth_dependency),
) -> FeedbackResponse:
    """Handle feedback requests.

    Args:
        feedback_request: The request containing feedback information.
        ensure_feedback_enabled: The feedback handler (FastAPI Depends) that
            will handle feedback status checks.
        auth: The Authentication handler (FastAPI Depends) that will
            handle authentication Logic.

    Returns:
        Response indicating the status of the feedback storage request.
    """
    logger.debug(f"feedback received {feedback_request}")

    user_id = retrieve_user_id(auth)
    try:
        store_feedback(user_id, feedback_request.model_dump(exclude=["model_config"]))
    except Exception as e:
        logger.error("Error storing user feedback")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "response": "Error storing user feedback",
                "cause": str(e),
            },
        )
        logger.error(e)

    return FeedbackResponse(response="feedback received")
