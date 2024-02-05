"""Data models representing payloads for REST API calls."""

from typing import Optional

from pydantic import BaseModel, model_validator


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
    conversation_id: Optional[str] = None
    response: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None

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

    @model_validator(mode="after")
    def validate_provider_and_model(self):
        """Perform validation on the provider and model."""
        if self.model and not self.provider:
            raise ValueError(
                "LLM provider must be specified when the model is specified!"
            )
        if self.provider and not self.model:
            raise ValueError(
                "LLM model must be specified when the provider is specified!"
            )
        return self


class FeedbackRequest(BaseModel):
    """Model representing a feedback request.

    Attributes:
        conversation_id: The required conversation ID.
        feedback_object: The JSON blob representing feedback.

    Example:
        ```python
        feedback_request = FeedbackRequest(
            conversation_id="12345678-abcd-0000-0123-456789abcdef",
            feedback_object='{"rating": 5, "comment": "Great service!"}'
        )
        ```
    """

    conversation_id: str
    feedback_object: str

    # provides examples for /docs endpoint
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "feedback_object": '{"rating": 5, "comment": "Great service!"}',
                    "conversation_id": "12345678-abcd-0000-0123-456789abcdef",
                }
            ]
        }
    }
