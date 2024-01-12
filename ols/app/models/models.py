from typing import Union

from pydantic import BaseModel


class LLMRequest(BaseModel):
    """Model representing a request for the LLM (Language Model).

    Attributes:
        query: The query string.
        conversation_id: The optional conversation ID.
        response: The optional response.

    Example:
        llm_request = LLMRequest(query="Tell me about Kubernetes")
    """

    query: str
    conversation_id: Union[str, None] = None
    response: Union[str, None] = None


class FeedbackRequest(BaseModel):
    """Model representing a feedback request.

    Attributes:
        conversation_id: The required conversation ID.
        feedback_object: The JSON blob representing feedback.

    Example:
        feedback_request = FeedbackRequest(conversation_id=123, feedback_object='{"rating": 5, "comment": "Great service!"}')
    """

    conversation_id: int
    feedback_object: str
