"""Unit tests for the API models."""

import pytest
from pydantic import ValidationError

from ols.app.models.models import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    LLMRequest,
    LLMResponse,
    ReferencedDocument,
    StatusResponse,
)
from ols.utils import suid


class TestLLM:
    """Unit tests for the LLMRequest/LLMResponse models."""

    @staticmethod
    def test_llm_request_required_inputs():
        """Test required inputs of the LLMRequest model."""
        query = "Tell me about Kubernetes"

        llm_request = LLMRequest(query=query)

        assert llm_request.query == query
        assert llm_request.conversation_id is None
        assert llm_request.provider is None
        assert llm_request.model is None

    @staticmethod
    def test_llm_request_optional_inputs():
        """Test optional inputs of the LLMRequest model."""
        query = "Tell me about Kubernetes"
        conversation_id = "id"
        provider = "openai"
        model = "gpt-3.5-turbo"
        llm_request = LLMRequest(
            query=query,
            conversation_id=conversation_id,
            provider=provider,
            model=model,
        )

        assert llm_request.query == query
        assert llm_request.conversation_id == conversation_id
        assert llm_request.provider == provider
        assert llm_request.model == model

    @staticmethod
    def test_llm_request_provider_and_model():
        """Test the LLMRequest model with provider and model."""
        # model set and provider not
        with pytest.raises(
            ValidationError,
            match="LLM provider must be specified when the model is specified.",
        ):
            LLMRequest(query="bla", provider=None, model="davinci")

        # provider set and model not
        with pytest.raises(
            ValidationError,
            match="LLM model must be specified when the provider is specified.",
        ):
            LLMRequest(query="bla", provider="openai", model=None)

    @staticmethod
    def test_llm_response():
        """Test the LLMResponse model."""
        conversation_id = "id"
        response = "response"
        referenced_documents = [
            ReferencedDocument(
                docs_url="https://foo.bar.com/index.html", title="Foo Bar"
            )
        ]

        llm_response = LLMResponse(
            conversation_id=conversation_id,
            response=response,
            referenced_documents=referenced_documents,
            truncated=False,
        )

        assert llm_response.conversation_id == conversation_id
        assert llm_response.response == response
        assert llm_response.referenced_documents == referenced_documents
        assert not llm_response.truncated


class TestStatusResponse:
    """Unit tests for the StatusResponse model."""

    @staticmethod
    def test_status_response():
        """Test the StatusResponse model."""
        functionality = "feedback"
        status = {"enabled": True}

        status_response = StatusResponse(functionality=functionality, status=status)

        assert status_response.functionality == functionality
        assert status_response.status == status


class TestFeedback:
    """Unit tests for the FeedbackRequest/FeedbackResponse models."""

    @staticmethod
    def test_feedback_request():
        """Test the FeedbackRequest model."""
        conversation_id = suid.get_suid()
        user_question = "user question"
        llm_response = "llm response"
        sentiment = 1
        user_feedback = "user feedback"

        feedback_request = FeedbackRequest(
            conversation_id=conversation_id,
            user_question=user_question,
            llm_response=llm_response,
            sentiment=sentiment,
            user_feedback=user_feedback,
        )

        assert feedback_request.conversation_id == conversation_id
        assert feedback_request.user_question == user_question
        assert feedback_request.llm_response == llm_response
        assert feedback_request.sentiment == sentiment
        assert feedback_request.user_feedback == user_feedback

    @staticmethod
    def test_feedback_request_optional_fields():
        """Test either sentiment or user_feedback needs to be set."""
        conversation_id = suid.get_suid()
        user_question = "user question"
        llm_response = "llm response"
        sentiment = 1
        user_feedback = "user feedback"

        # no sentiment or user_feedback raises validation error
        with pytest.raises(
            ValidationError, match="Either 'sentiment' or 'user_feedback' must be set"
        ):
            FeedbackRequest(
                conversation_id=conversation_id,
                user_question=user_question,
                llm_response=llm_response,
            )

        # just of those set doesn't raise - sentiment set
        FeedbackRequest(
            conversation_id=conversation_id,
            user_question=user_question,
            llm_response=llm_response,
            sentiment=sentiment,
        )

        # just of those set doesn't raise - user_feedback set
        FeedbackRequest(
            conversation_id=conversation_id,
            user_question=user_question,
            llm_response=llm_response,
            user_feedback=user_feedback,
        )

    @staticmethod
    def test_feedback_request_improper_conversation_id():
        """Test if conversation ID format is checked."""
        conversation_id = "this-is-bad"
        user_question = "user question"
        llm_response = "llm response"
        sentiment = 1
        user_feedback = "user feedback"

        # ValueError should be raised
        with pytest.raises(ValueError, match="Improper conversation ID this-is-bad"):
            FeedbackRequest(
                conversation_id=conversation_id,
                user_question=user_question,
                llm_response=llm_response,
                sentiment=sentiment,
                user_feedback=user_feedback,
            )

    @staticmethod
    def test_feedback_response():
        """Test the FeedbackResponse model."""
        feedback_response = "feedback received"

        feedback_request = FeedbackResponse(response=feedback_response)

        assert feedback_request.response == feedback_response


class TestHealth:
    """Unit test for the HealthResponse model."""

    @staticmethod
    def test_health_response():
        """Test the HealthResponse model."""
        health_response = {"status": "healthy"}

        health_request = HealthResponse(status=health_response)

        assert health_request.status == health_response
