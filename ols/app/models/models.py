"""Data models representing payloads for REST API calls."""

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
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "write a deployment yaml for the mongodb image",
                    "conversation_id": "",
                    "response": "",
                }
            ]
        }
    }


class FeedbackRequest(BaseModel):
    """Model representing a feedback request.

    Attributes:
        conversation_id: The required conversation ID.
        feedback_object: The JSON blob representing feedback.

    Example:
        ```python
        feedback_request = FeedbackRequest(
            conversation_id=123,
            feedback_object='{"rating": 5, "comment": "Great service!"}'
        )
        ```
    """

    conversation_id: int
    feedback_object: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "feedback_object": '{"rating": 5, "comment": "Great service!"}',
                    "conversation_id": "1234",
                }
            ]
        }
    }
