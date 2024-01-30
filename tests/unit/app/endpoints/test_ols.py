"""Unit tests for OLS endpoint."""

import logging

import pytest
from fastapi import HTTPException, status

from ols.app.endpoints.ols import verify_request_provider_and_model
from ols.app.models.models import LLMRequest


class TestVerifyRequestProviderAndModel:
    """Test the verify_request_provider_and_model function."""

    def test_provider_set_model_not_raises(self):
        """Test raise when the provider is set and the model is not."""
        request = LLMRequest(query="bla", provider="provider", model=None)
        with pytest.raises(
            HTTPException,
            match="LLM model must be specified when provider is specified",
        ) as e:
            verify_request_provider_and_model(request)
            assert e.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_model_set_provider_not_raises(self):
        """Test no raise when the model is set and the provider is not."""
        request = LLMRequest(query="bla", provider=None, model="model")
        with pytest.raises(
            HTTPException,
            match="LLM provider must be specified when the model is specified",
        ) as e:
            verify_request_provider_and_model(request)
            assert e.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_model_and_provider_logged_when_set(self, caplog):
        """Test that the function logs the provider and model when they are set."""
        caplog.set_level(logging.DEBUG)
        request = LLMRequest(query="bla", provider="provider", model="model")
        verify_request_provider_and_model(request)

        # check captured outputs
        captured_out = caplog.text
        assert "provider 'provider' is set in request" in captured_out
        assert "model 'model' is set in request" in captured_out

    def test_nothing_is_logged_when_provider_and_model_not_set(self, caplog):
        """Test that the function does not log anything when the provider and model are not set."""
        request = LLMRequest(query="bla", provider=None, model=None)
        verify_request_provider_and_model(request)

        # check captured outputs
        captured_out = caplog.text
        assert captured_out == ""
