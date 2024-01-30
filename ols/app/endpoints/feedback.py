"""Handler for REST API call to provide user feedback."""

import logging

from fastapi import APIRouter

from ols.app.models.models import FeedbackRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("")
def user_feedback(feedback_request: FeedbackRequest) -> dict:
    """Handle feedback requests.

    Args:
        feedback_request: The request containing feedback information.

    Returns:
        Response indicating the status of the feedback.
    """
    conversation = feedback_request.conversation_id
    logger.info(f"{conversation} New feedback received")
    logger.info(f"{conversation} Feedback blob: {feedback_request.feedback_object}")

    return {"status": "feedback received"}
