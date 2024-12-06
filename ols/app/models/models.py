"""Data models representing payloads for REST API calls."""

from collections import OrderedDict
from typing import Optional, Self

from pydantic import BaseModel, field_validator, model_validator
from pydantic.dataclasses import dataclass

from ols.utils import suid


class Attachment(BaseModel):
    """Model representing an attachment that can be send from UI as part of query.

    List of attachments can be optional part of 'query' request.

    Attributes:
        attachment_type: The attachment type, like "log", "configuration" etc.
        content_type: The content type as defined in MIME standard
        content: The actual attachment content

    YAML attachments with **kind** and **metadata/name** attributes will
    be handled as resources with specified name:
    ```
    kind: Pod
    metadata:
        name: private-reg
    ```
    """

    attachment_type: str
    content_type: str
    content: str

    # provides examples for /docs endpoint
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "attachment_type": "log",
                    "content_type": "text/plain",
                    "content": "this is attachment",
                },
                {
                    "attachment_type": "configuration",
                    "content_type": "application/yaml",
                    "content": "kind: Pod\n metadata:\n name:    private-reg",
                },
                {
                    "attachment_type": "configuration",
                    "content_type": "application/yaml",
                    "content": "foo: bar",
                },
            ]
        }
    }


class LLMRequest(BaseModel):
    """Model representing a request for the LLM (Language Model) send into OLS service.

    Attributes:
        query: The query string.
        conversation_id: The optional conversation ID (UUID).
        provider: The optional provider.
        model: The optional model.
        attachments: The optional attachments.

    Example:
        ```python
        llm_request = LLMRequest(query="Tell me about Kubernetes")
        ```
    """

    query: str
    conversation_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    attachments: Optional[list[Attachment]] = None

    # provides examples for /docs endpoint
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "query": "write a deployment yaml for the mongodb image",
                    "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "attachments": [
                        {
                            "attachment_type": "log",
                            "content_type": "text/plain",
                            "content": "this is attachment",
                        },
                        {
                            "attachment_type": "configuration",
                            "content_type": "application/yaml",
                            "content": "kind: Pod\n metadata:\n    name: private-reg",
                        },
                        {
                            "attachment_type": "configuration",
                            "content_type": "application/yaml",
                            "content": "foo: bar",
                        },
                    ],
                }
            ]
        },
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


@dataclass(frozen=True, unsafe_hash=False)
class ReferencedDocument:
    """RAG referenced document.

    Attributes:
    docs_url: URL of the corresponding OCP documentation page
    title: Title of the corresponding OCP documentation page
    """

    docs_url: str
    title: str

    @staticmethod
    def from_rag_chunks(rag_chunks: list["RagChunk"]) -> list["ReferencedDocument"]:
        """Create a list of ReferencedDocument from a list of rag_chunks.

        Order of items is preserved.
        """
        return list(
            OrderedDict(
                (
                    rag_chunk.doc_url,
                    ReferencedDocument(rag_chunk.doc_url, rag_chunk.doc_title),
                )
                for rag_chunk in rag_chunks
            ).values()
        )


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


class NotAvailableResponse(BaseModel):
    """Model representing error response for readiness endpoint."""

    detail: dict[str, str]

    # provides examples for /docs endpoint
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "detail": {
                        "response": "Service is not ready",
                        "cause": "Index is not ready",
                    }
                },
                {
                    "detail": {
                        "response": "Service is not ready",
                        "cause": "LLM is not ready",
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
        if not suid.check_suid(value):
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
    def check_sentiment_or_user_feedback_set(self) -> Self:
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


class LivenessResponse(BaseModel):
    """Model representing a response to a liveness request.

    Attributes:
        alive: If app is alive.

    Example:
        ```python
        liveness_response = LivenessResponse(alive=True)
        ```
    """

    alive: bool

    # provides examples for /docs endpoint
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "alive": True,
                }
            ]
        }
    }


class ReadinessResponse(BaseModel):
    """Model representing a response to a readiness request.

    Attributes:
        ready: The readiness of the service.
        reason: The reason for the readiness.

    Example:
        ```python
        readiness_response = ReadinessResponse(ready=True, reason="service is ready")
        ```
    """

    ready: bool
    reason: str

    # provides examples for /docs endpoint
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ready": True,
                    "reason": "service is ready",
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


@dataclass
class RagChunk:
    """Model representing a RAG chunk.

    Attributes:
        text: The text used as a RAG chunk.
        doc_url: The URL of the doc from which the RAG chunk comes from.
        doc_title: The title of the doc.
    """

    text: str
    doc_url: str
    doc_title: str


@dataclass
class SummarizerResponse:
    """Model representing a response from the summarizer.

    Attributes:
        response: The response from the summarizer.
        rag_chunks: The RAG chunks.
        history_truncated: Whether the history was truncated.
    """

    response: str
    rag_chunks: list[RagChunk]
    history_truncated: bool


class CacheEntry(BaseModel):
    """Model representing a cache entry.

    Attributes:
        query: The query string.
        response: The response string.
    """

    query: str
    response: Optional[str] = ""
    attachments: list[Attachment] = []

    @field_validator("response")
    @classmethod
    def set_none_response_to_empty_string(cls, v: Optional[str]) -> str:
        """Convert None response to an empty string."""
        if v is None:
            return ""
        return v

    def to_dict(self) -> dict:
        """Convert the cache entry to a dictionary."""
        return {
            "human_query": self.query,
            "ai_response": self.response,
            "attachments": [attachment.model_dump() for attachment in self.attachments],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create a cache entry from a dictionary."""
        return cls(
            query=data["human_query"],
            response=data["ai_response"],
            attachments=[
                Attachment(**attachment) for attachment in data["attachments"]
            ],
        )

    @staticmethod
    def cache_entries_to_history(cache_entries: list["CacheEntry"]) -> list[str]:
        """Convert cache entries to a history."""
        history = []
        for entry in cache_entries:
            history.append(f"human: {entry.query.strip()}")
            # the real response or empty string when response is not recorded
            history.append(f"ai: {str(entry.response).strip()}")

        return history
