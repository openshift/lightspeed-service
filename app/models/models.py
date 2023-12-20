from typing import Union

from pydantic import BaseModel


class LLMRequest(BaseModel):
    """
    Model representing a request for the LLM (Language Model).

    Attributes:
        query (str): The query string.
        conversation_id (Union[str, None]): The optional conversation ID.
        response (Union[str, None]): The optional response.

    Example:
        llm_request = LLMRequest(query="Tell me about Kubernetes")
    """

    query: str
    conversation_id: Union[str, None] = None
    response: Union[str, None] = None


class FeedbackRequest(BaseModel):
    """
    Model representing a feedback request.

    Attributes:
        conversation_id (int): The required conversation ID.
        feedback_object (str): The JSON blob representing feedback.

    Example:
        feedback_request = FeedbackRequest(conversation_id=123, feedback_object='{"rating": 5, "comment": "Great service!"}')
    """

    conversation_id: int
    feedback_object: str
