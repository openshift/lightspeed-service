"""Handler for REST API call to provide user feedback."""

import logging

from fastapi import APIRouter, Depends

from ols.app.models.models import FeedbackRequest, FeedbackResponse
from ols.utils.auth_dependency import auth_dependency

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("")
def user_feedback(
    feedback_request: FeedbackRequest, auth=Depends(auth_dependency)
) -> FeedbackResponse:
    """Handle feedback requests.

    Args:
        feedback_request: The request containing feedback information.
        auth: The Authentication handler (FastAPI Depends) that will handle authentication Logic.

    Returns:
        Response indicating the status of the feedback.
    """
    logger.info(
        f"feed back received for conversation '{feedback_request.conversation_id}', "
        f"feedback blob: {feedback_request.feedback_object}"
    )

    return FeedbackResponse(response="feedback received")
