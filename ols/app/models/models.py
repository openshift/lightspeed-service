"""Data models representing payloads for REST API calls."""

from typing import Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class LLMRequest(BaseModel):
    """Model representing a request for the LLM (Language Model).

    Attributes:
        query: The query string.
        conversation_id: The optional conversation id in valid UUID format.
        response: The optional response.

    Example:
        llm_request = LLMRequest(query="Tell me about Kubernetes")
    """

    query: str
    conversation_id: UUID = Field(default_factory=uuid4)
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
