"""Data models representing payloads for REST API calls."""

from typing import Any, Dict, Optional, Self

from pydantic import BaseModel, field_validator, model_validator

from ols.utils import suid


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


class ReferencedDocument(BaseModel):
    """RAG referenced document.

    Attributes:
    docs_url: URL of the corresponding OCP documentation page
    title: Title of the corresponding OCP documentation page
    """

    docs_url: Optional[str] = None
    title: Optional[str] = None

    def __init__(self, docs_url: str, title: str) -> None:
        """Initialize a ReferencedDocument."""
        super().__init__()
        self.docs_url = docs_url
        self.title = title

    def __eq__(self, other: object) -> bool:
        """Compare two objects for equality."""
        if isinstance(other, ReferencedDocument):
            return self.docs_url == other.docs_url and self.title == other.title
        return False

    @staticmethod
    def json_decode_object_hook(dct: Dict[str, Any]) -> Any:
        """Deserialize dict into ReferencedDocument if we can."""
        if "docs_url" in dct and "title" in dct:
            return ReferencedDocument(**dct)
        return dct


class LLMResponse(BaseModel):
    """Model representing a response from the LLM (Language Model).

    Attributes:
        conversation_id: The optional conversation ID (UUID).
        response: The optional response.
        referenced_documents: The optional URLs and titles for the documents used
                              to generate the response.
        truncated: Set to True if conversation history was truncated to be within context window.
    """

    conversation_id: str
    response: str
    referenced_documents: list[ReferencedDocument]
    truncated: bool

    # provides examples for /docs endpoint
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
                    "response": "Operator Lifecycle Manager (OLM) helps users install...",
                    "referenced_documents": [
                        {
                            "docs_url": "https://docs.openshift.com/container-platform/4.15/operators/"
                            "understanding/olm/olm-understanding-olm.html",
                            "title": "Operator Lifecycle Manager concepts and resources",
                        },
                    ],
                    "truncated": False,
                }
            ]
        }
    }


class UnauthorizedResponse(BaseModel):
    """Model representing response for missing or invalid credentials."""

    detail: str

    # provides examples for /docs endpoint
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "detail": "Unauthorized: No auth header found",
                },
            ]
        }
    }


class ForbiddenResponse(BaseModel):
    """Model representing response when client does not have permission to access resource."""

    detail: str

    # provides examples for /docs endpoint
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "detail": "Unable to review token",
                },
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Model representing error response for query endpoint."""

    detail: dict[str, str]

    # provides examples for /docs endpoint
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "detail": {
                        "response": "Error while validation question",
                        "cause": "Failed to handle request to https://bam-api.res.ibm.com/v2/text",
                    },
                },
                {
                    "detail": {
                        "response": "Error retrieving conversation history",
                        "cause": "Invalid conversation ID 1237-e89b-12d3-a456-426614174000",
                    },
                },
            ]
        }
    }


class PromptTooLongResponse(ErrorResponse):
    """Model representing error response when prompt is too long."""

    # provides examples for /docs endpoint
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "detail": {
                        "response": "Prompt is too long",
                        "cause": "Prompt length exceeds LLM context window limit (8000 tokens)",
                    },
                },
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
            user_feedback="Great service!",
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
                    "user_question": "foo",
                    "llm_response": "bar",
                    "user_feedback": "Great service!",
                    "sentiment": 1,
                }
            ]
        }
    }

    @field_validator("conversation_id")
    @classmethod
    def check_uuid(cls, value: str) -> str:
        """Check if conversation ID has the proper format."""
        if not (suid.check_suid(value)):
            raise ValueError(f"Improper conversation ID {value}")
        return value

    @field_validator("sentiment")
    @classmethod
    def check_sentiment(cls, value: Optional[int]) -> Optional[int]:
        """Check if sentiment value is as expected."""
        if value not in {-1, 1, None}:
            raise ValueError(f"Improper value {value}, needs to be -1 or 1")
        return value

    @model_validator(mode="after")
    def check_sentiment_or_user_feedback_set(self) -> "FeedbackRequest":
        """Ensure that either 'sentiment' or 'user_feedback' is set."""
        if self.sentiment is None and self.user_feedback is None:
            raise ValueError("Either 'sentiment' or 'user_feedback' must be set")
        return self


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
