"""Data models representing payloads for REST API calls."""

from pydantic import BaseModel


class LLMRequest(BaseModel):
    """Model representing a request for the LLM (Language Model).

    Attributes:
        query: The query string.
        conversation_id: The optional conversation ID.
        response: The optional response.
        provider: The optional provider.
        model: The optional model.

    Example:
        llm_request = LLMRequest(query="Tell me about Kubernetes")
    """

    query: str
    conversation_id: str | None = None
    response: str | None = None
    provider: str | None = None
    model: str | None = None

    # provides examples for /docs endpoint
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

    # provides examples for /docs endpoint
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
