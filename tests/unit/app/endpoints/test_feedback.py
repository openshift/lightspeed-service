"""Unit tests for feedback endpoint handlers."""

import pytest

from ols.app.endpoints import feedback
from ols.app.models.models import FeedbackRequest
from ols.utils import config, suid


@pytest.fixture(scope="module")
def load_config():
    """Load config before unit tests."""
    config.init_config("tests/config/test_app_endpoints.yaml")


def test_user_feedback(load_config):
    """Test user feedback API endpoint."""
    feedback_request = FeedbackRequest(
        conversation_id=suid.get_suid(),
        feedback_object='{"rating": 5, "comment": "Great service!"}',
    )
    response = feedback.user_feedback(feedback_request)
    assert response == {"status": "feedback received"}
