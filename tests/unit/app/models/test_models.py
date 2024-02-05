"""Unit tests for the API models."""

import pytest
from pydantic import ValidationError

from ols.app.models.models import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    LLMRequest,
    LLMResponse,
)


class TestLLM:
    """Unit tests for the LLMRequest/LLMResponse models."""

    def test_llm_request_required_inputs(self):
        """Test required inputs of the LLMRequest model."""
        query = "Tell me about Kubernetes"

        llm_request = LLMRequest(query=query)

        assert llm_request.query == query
        assert llm_request.conversation_id is None
        assert llm_request.provider is None
        assert llm_request.model is None

    def test_llm_request_optional_inputs(self):
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

    def test_llm_request_provider_and_model(self):
        """Test the LLMRequest model with provider and model."""
        # model set and provider not
        with pytest.raises(
            ValidationError,
            match="LLM provider must be specified when the model is specified!",
        ):
            LLMRequest(query="bla", provider=None, model="davinci")

        # provider set and model not
        with pytest.raises(
            ValidationError,
            match="LLM model must be specified when the provider is specified!",
        ):
            LLMRequest(query="bla", provider="openai", model=None)

    def test_llm_response(self):
        """Test the LLMResponse model."""
        conversation_id = "id"
        response = "response"

        llm_response = LLMResponse(
            conversation_id=conversation_id,
            response=response,
        )

        assert llm_response.conversation_id == conversation_id
        assert llm_response.response == response


class TestFeedback:
    """Unit tests for the FeedbackRequest/FeedbackResponse models."""

    def test_feedback_request(self):
        """Test the FeedbackRequest model."""
        conversation_id = "id"
        feedback_obj = {"rating": 5, "comment": "Great service!"}

        feedback_request = FeedbackRequest(
            conversation_id=conversation_id,
            feedback_object=feedback_obj,
        )

        assert feedback_request.conversation_id == conversation_id
        assert feedback_request.feedback_object == feedback_obj

    def test_feedback_response(self):
        """Test the FeedbackResponse model."""
        feedback_response = "feedback received"

        feedback_request = FeedbackResponse(response=feedback_response)

        assert feedback_request.response == feedback_response


class TestHealth:
    """Unit test for the HealthResponse model."""

    def test_health_response(self):
        """Test the HealthResponse model."""
        health_response = {"status": "healthy"}

        health_request = HealthResponse(status=health_response)

        assert health_request.status == health_response
