"""Handler for REST API call to provide user feedback."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ols import config
from ols.app.endpoints.ols import retrieve_user_id
from ols.app.models.models import (
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    ForbiddenResponse,
    StatusResponse,
    UnauthorizedResponse,
)
from ols.src.auth.auth import get_auth_dependency
from ols.utils.suid import get_suid

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/feedback", tags=["feedback"])
auth_dependency = get_auth_dependency(config.ols_config, virtual_path="/ols-access")


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
            detail="Forbidden: Feedback is currently disabled.",
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
    # Creates storage path only if it doesn't exist. The `exist_ok=True` prevents
    # race conditions in case of multiple server instances trying to set up storage
    # at the same location.
    storage_path = Path(config.ols_config.user_data_collection.feedback_storage)
    storage_path.mkdir(parents=True, exist_ok=True)

    current_time = str(datetime.utcnow())
    data_to_store = {"user_id": user_id, "timestamp": current_time, **feedback}

    # stores feedback in a file under unique uuid
    feedback_file_path = storage_path / f"{get_suid()}.json"
    with open(feedback_file_path, "w", encoding="utf-8") as feedback_file:
        json.dump(data_to_store, feedback_file)

    logger.debug("feedback stored in '%s'", feedback_file_path)


@router.get("/status")
def feedback_status() -> StatusResponse:
    """Handle feedback status requests.

    Returns:
        Response indicating the status of the feedback.
    """
    logger.debug("feedback status request received")
    feedback_status_enabled = is_feedback_enabled()
    return StatusResponse(
        functionality="feedback", status={"enabled": feedback_status_enabled}
    )


post_feedback_responses: dict[int | str, dict[str, Any]] = {
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
    ensure_feedback_enabled: Annotated[Any, Depends(ensure_feedback_enabled)],
    auth: Annotated[Any, Depends(auth_dependency)],
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
    logger.debug("feedback received %s", str(feedback_request))

    user_id = retrieve_user_id(auth)
    try:
        store_feedback(user_id, feedback_request.model_dump(exclude={"model_config"}))
    except Exception as e:
        logger.error("Error storing user feedback: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "response": "Error storing user feedback",
                "cause": str(e),
            },
        )

    return FeedbackResponse(response="feedback received")
