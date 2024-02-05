"""Unit tests for feedback endpoint handlers."""

from ols.app.endpoints import feedback
from ols.app.models.models import FeedbackRequest, FeedbackResponse
from ols.utils import suid


def test_user_feedback():
    """Test user feedback API endpoint."""
    feedback_request = FeedbackRequest(
        conversation_id=suid.get_suid(),
        feedback_object={"rating": 5, "comment": "Great service!"},
    )

    response = feedback.user_feedback(feedback_request)

    assert response == FeedbackResponse(response="feedback received")
