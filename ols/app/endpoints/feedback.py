from fastapi import APIRouter

from ols.app.models.models import FeedbackRequest
from ols.utils import config

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("")
def feedback_request(feedback_request: FeedbackRequest) -> dict:
    """Handle feedback requests.

    Args:
        feedback_request: The request containing feedback information.

    Returns:
        Response indicating the status of the feedback.
    """
    conversation = str(feedback_request.conversation_id)
    config.default_logger.info(f"{conversation} New feedback received")
    config.default_logger.info(
        f"{conversation} Feedback blob: {feedback_request.feedback_object}"
    )

    return {"status": "feedback received"}
