from fastapi import APIRouter

from app.models.models import FeedbackRequest
from utils.logger import Logger

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("")
def feedback_request(feedback_request: FeedbackRequest):
    """
    Handle feedback requests.

    Args:
        feedback_request (FeedbackRequest): The request containing feedback information.

    Returns:
        dict: Response indicating the status of the feedback.
    """
    logger = Logger("feedback_endpoint").logger

    conversation = str(feedback_request.conversation_id)
    logger.info(f"{conversation} New feedback received")
    logger.info(f"{conversation} Feedback blob: {feedback_request.feedback_object}")

    return {"status": "feedback received"}
