"""Handler for REST API call to provide user feedback."""

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ols.app.models.models import (
    FeedbackRequest,
    FeedbackResponse,
    FeedbacksListResponse,
    StatusResponse,
)
from ols.utils import config
from ols.utils.auth_dependency import auth_dependency
from ols.utils.suid import get_suid

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/feedback", tags=["feedback"])


async def feedback_is_enabled(request: Request) -> bool:
    """Check if feedback is enabled.

    Args:
        request (Request): The FastAPI request object.

    Returns:
        True if feedback is enabled, False otherwise.

    Raises:
        HTTPException: If feedback is disabled.
    """
    feedback_enabled = not config.ols_config.user_data_collection.feedback_disabled
    if not feedback_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Feedback is currently disabled.",
        )
    return feedback_enabled


def get_feedback_status() -> bool:
    """Check if feedback is enabled.

    Returns:
        True if feedback is enabled, False otherwise.
    """
    return not config.ols_config.user_data_collection.feedback_disabled


def list_feedbacks() -> list[str]:
    """List feedbacks in the local filesystem.

    Returns:
        List of feedback files (without the ".json" extension).
    """
    storage_path = Path(config.ols_config.user_data_collection.feedback_storage)
    feedback_files = list(storage_path.glob("*.json"))
    # extensions are trimmed, eg. ["12345678-abcd-0000-0123-456789abcdef", ...]
    feedbacks = [f.stem for f in feedback_files]
    logger.info(f"'{len(feedbacks)}' feedbacks found")
    return feedbacks


def store_feedback(user_id: str, feedback: dict) -> None:
    """Store feedback in the local filesystem.

    Args:
        user_id: The user ID (UUID).
        feedback: The feedback to store.
    """
    # ensures storage path exists
    storage_path = Path(config.ols_config.user_data_collection.feedback_storage)
    if not storage_path.exists():
        storage_path.mkdir(parents=True)

    data_to_store = {"user_id": user_id, **feedback}

    # stores feedback in a file under unique uuid
    feedback_file_path = storage_path / f"{get_suid()}.json"
    with open(feedback_file_path, "w", encoding="utf-8") as feedback_file:
        json.dump(data_to_store, feedback_file)

    logger.info(f"feedback stored in '{feedback_file_path}'")


def remove_feedback(feedback_id: str) -> None:
    """Remove feedback from the local filesystem.

    Args:
        feedback_id: The feedback ID (UUID).
    """
    storage_path = Path(config.ols_config.user_data_collection.feedback_storage)
    feedback_file = storage_path / f"{feedback_id}.json"
    if feedback_file.exists():
        feedback_file.unlink()
        if not feedback_file.exists():
            logger.info(f"feedback file '{feedback_file}' removed")
        else:
            logger.error(f"feedback file '{feedback_file}' failed to remove")
    else:
        logger.error(f"feedback file '{feedback_file}' not found")


@router.get("/status")
def feedback_status() -> StatusResponse:
    """Handle feedback status requests.

    Returns:
        Response indicating the status of the feedback.
    """
    logger.info("feedback status request received")
    feedback_status = get_feedback_status()
    return StatusResponse(functionality="feedback", status={"enabled": feedback_status})


# TODO: OLS-136 implements the collection mechanism - revisit the need
# of this endpoint
# If endpoint stays in place, it needs to be properly secured - OLS-404
@router.get("/list")
def get_user_feedbacks(
    feedback_is_enabled: Any = Depends(feedback_is_enabled),
    auth: Any = Depends(auth_dependency),
) -> FeedbacksListResponse:
    """Handle feedback listing requests.

    Args:
        feedback_is_enabled: The feedback handler (FastAPI Depends) that
            will handle feedback status checks.
        auth: The Authentication handler (FastAPI Depends) that will
            handle authentication Logic.

    Returns:
        Response containing the list of feedbacks.
    """
    logger.info("feedback list request received")

    feedbacks = list_feedbacks()

    return FeedbacksListResponse(feedbacks=feedbacks)


@router.post("")
def store_user_feedback(
    feedback_request: FeedbackRequest,
    feedback_is_enabled: Any = Depends(feedback_is_enabled),
    auth: Any = Depends(auth_dependency),
) -> FeedbackResponse:
    """Handle feedback requests.

    Args:
        feedback_request: The request containing feedback information.
        feedback_is_enabled: The feedback handler (FastAPI Depends) that
            will handle feedback status checks.
        auth: The Authentication handler (FastAPI Depends) that will
            handle authentication Logic.

    Returns:
        Response indicating the status of the feedback storage request.
    """
    logger.info(f"feedback received {feedback_request}")

    user_id = auth[0]
    store_feedback(user_id, feedback_request.model_dump(exclude=["model_config"]))

    return FeedbackResponse(response="feedback received")


# TODO: OLS-136 implements the collection mechanism - revisit the need
# of this endpoint
# If endpoint stays in place, it needs to be properly secured - OLS-404
@router.delete("/{feedback_id}")
def remove_user_feedback(
    feedback_id: str,
    feedback_is_enabled: Any = Depends(feedback_is_enabled),
    auth: Any = Depends(auth_dependency),
) -> FeedbackResponse:
    """Handle feedback removal requests.

    Args:
        feedback_id: The feedback ID (UUID) to be removed.
        feedback_is_enabled: The feedback handler (FastAPI Depends) that
            will handle feedback status checks.
        auth: The Authentication handler (FastAPI Depends) that will
            handle authentication Logic.

    Returns:
        Response indicating the status of the feedback removal.
    """
    logger.info(f"feedback '{feedback_id}' removal request received")
    remove_feedback(feedback_id)

    return FeedbackResponse(response="feedback removed")
