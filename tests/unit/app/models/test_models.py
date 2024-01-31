"""Unit tests for the API models."""

from ols.app.models.models import FeedbackRequest, LLMRequest
from ols.utils import suid


def test_feedback_request():
    """Test the FeedbackRequest model."""
    conversation_id = suid.get_suid()

    feedback_request = FeedbackRequest(
        conversation_id=conversation_id,
        feedback_object='{"rating": 5, "comment": "Great service!"}',
    )
    assert feedback_request.conversation_id == conversation_id
    assert (
        feedback_request.feedback_object == '{"rating": 5, "comment": "Great service!"}'
    )


def test_llm_request():
    """Test the LLMRequest model."""
    llm_request = LLMRequest(query="Tell me about Kubernetes")
    assert llm_request.query == "Tell me about Kubernetes"
    assert llm_request.conversation_id is None
    assert llm_request.response is None

    llm_request = LLMRequest(
        query="Tell me about Kubernetes",
        conversation_id="abc",
        response="Kubernetes is a portable, extensible, open source platform ...",
    )
    assert llm_request.query == "Tell me about Kubernetes"
    assert llm_request.conversation_id == "abc"
    assert (
        llm_request.response
        == "Kubernetes is a portable, extensible, open source platform ..."
    )
