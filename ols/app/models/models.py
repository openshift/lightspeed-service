"""Data models representing payloads for REST API calls."""

from typing import Optional, Self

from pydantic import BaseModel, model_validator


class LLMRequest(BaseModel):
    """Model representing a request for the LLM (Language Model).

    Attributes:
        query: The query string.
        conversation_id: The optional conversation ID (UUID).
        provider: The optional provider.
        model: The optional model.

    Example:
        ```python
        llm_request = LLMRequest(query="Tell me about Kubernetes")
        ```
    """

    query: str
    conversation_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None

    # provides examples for /docs endpoint
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "write a deployment yaml for the mongodb image",
                    "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                }
            ]
        }
    }

    @model_validator(mode="after")
    def validate_provider_and_model(self) -> Self:
        """Perform validation on the provider and model."""
        if self.model and not self.provider:
            raise ValueError(
                "LLM provider must be specified when the model is specified."
            )
        if self.provider and not self.model:
            raise ValueError(
                "LLM model must be specified when the provider is specified."
            )
        return self


class LLMResponse(BaseModel):
    """Model representing a response from the LLM (Language Model).

    Attributes:
        conversation_id: The optional conversation ID (UUID).
        response: The optional response.
        referenced_documents: The optional URLs for the documents used to generate the response.
        truncated: Set to True if conversation history was truncated to be within context window.
    """

    conversation_id: str
    response: str
    referenced_documents: list[str]
    truncated: bool

    # provides examples for /docs endpoint
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
                    "response": "Operator Lifecycle Manager (OLM) helps users install...",
                    "referenced_documents": [
                        "https://docs.openshift.com/container-platform/4.14/operators/"
                        "understanding/olm/olm-understanding-olm.html"
                    ],
                }
            ]
        }
    }


class StatusResponse(BaseModel):
    """Model representing a response to a status request.

    Attributes:
        functionality: The functionality of the service.
        status: The status of the service.

    Example:
        ```python
        status_response = StatusResponse(
            functionality="feedback",
            status={"enabled": True},
        )
        ```
    """

    functionality: str
    status: dict

    # provides examples for /docs endpoint
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "functionality": "feedback",
                    "status": {"enabled": True},
                }
            ]
        }
    }


class FeedbackRequest(BaseModel):
    """Model representing a feedback request.

    Attributes:
        conversation_id: The required conversation ID (UUID).
        user_question: The required user question.
        llm_response: The required LLM response.
        sentiment: The optional sentiment.
        user_feedback: The optional user feedback.

    Example:
        ```python
        feedback_request = FeedbackRequest(
            conversation_id="12345678-abcd-0000-0123-456789abcdef",
            user_question="what are you doing?",
            llm_response="I don't know",
            sentiment=-1,
        )
        ```
    """

    conversation_id: str
    user_question: str
    llm_response: str
    sentiment: Optional[int] = None
    user_feedback: Optional[str] = None

    # provides examples for /docs endpoint
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "conversation_id": "12345678-abcd-0000-0123-456789abcdef",
                    "feedback_object": {"rating": 5, "comment": "Great service!"},
                }
            ]
        }
    }

    @model_validator(mode="after")
    def check_sentiment_or_user_feedback_set(self) -> "FeedbackRequest":
        """Ensure that either 'sentiment' or 'user_feedback' is set."""
        if self.sentiment is None and self.user_feedback is None:
            raise ValueError("Either 'sentiment' or 'user_feedback' must be set")
        return self


class FeedbacksListResponse(BaseModel):
    """Model representing a response to a feedback list request.

    Attributes:
        feedbacks: The list of feedback IDs.

    Example:
        ```python
        feedbacks_list_response = FeedbacksListResponse(
            feedbacks=["12345678-abcd-0000-0123-456789abcdef"]
        )
        ```
    """

    feedbacks: list[str]

    # provides examples for /docs endpoint
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "feedbacks": ["12345678-abcd-0000-0123-456789abcdef"],
                }
            ]
        }
    }


class FeedbackResponse(BaseModel):
    """Model representing a response to a feedback request.

    Attributes:
        response: The response of the feedback request.

    Example:
        ```python
        feedback_response = FeedbackResponse(response="feedback received")
        ```
    """

    response: str

    # provides examples for /docs endpoint
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "response": "feedback received",
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Model representing a response to a health request.

    Attributes:
        status: The status of the app.

    Example:
        ```python
        health_response = HealthResponse(status={"status": "healthy"})
        ```
    """

    status: dict[str, str]

    # provides examples for /docs endpoint
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": {"status": "healthy"},
                }
            ]
        }
    }


class AuthorizationResponse(BaseModel):
    """Model representing a response to an authorization request.

    Attributes:
        user_id: The ID of the logged in user.
        username: The name of the logged in user.
    """

    user_id: str
    username: str

    # provides examples for /docs endpoint
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "123e4567-e89b-12d3-a456-426614174000",
                    "username": "user1",
                }
            ]
        }
    }
