"""Handler for REST API call to provide user feedback."""

import logging

from fastapi import APIRouter

from ols.app.models.models import FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("")
def user_feedback(feedback_request: FeedbackRequest) -> FeedbackResponse:
    """Handle feedback requests.

    Args:
        feedback_request: The request containing feedback information.

    Returns:
        Response indicating the status of the feedback.
    """
    logger.info(
        f"feed back received for conversation '{feedback_request.conversation_id}', "
        f"feedback blob: {feedback_request.feedback_object}"
    )

    return FeedbackResponse(response="feedback received")
