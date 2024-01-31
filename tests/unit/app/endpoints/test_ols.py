"""Unit tests for OLS endpoint."""

import logging
from http import HTTPStatus
from unittest.mock import patch

import pytest
from fastapi import HTTPException, status

from ols import constants
from ols.app.endpoints import ols
from ols.app.endpoints.ols import verify_request_provider_and_model
from ols.app.models.models import LLMRequest
from ols.app.utils import Utils
from ols.utils import config
from tests.mock_classes.llm_chain import mock_llm_chain
from tests.mock_classes.llm_loader import mock_llm_loader


@pytest.fixture(scope="module")
def load_config():
    """Load config before unit tests."""
    config.init_config("tests/config/test_app_endpoints.yaml")


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


@patch("ols.src.query_helpers.question_validator.QuestionValidator.validate_question")
@patch("ols.src.query_helpers.docs_summarizer.DocsSummarizer.summarize")
@patch("ols.src.query_helpers.yaml_generator.YamlGenerator.generate_yaml")
@patch("ols.utils.config.conversation_cache.get")
def test_conversation_request(
    mock_conversation_cache_get,
    mock_generate_yaml,
    mock_summarize,
    mock_validate_question,
    load_config,
):
    """Test conversation request API endpoint."""
    # valid question, no yaml
    mock_validate_question.return_value = [
        constants.SUBJECT_VALID,
        constants.CATEGORY_GENERIC,
    ]
    mock_summarize.return_value = (
        "Kubernetes is an open-source container-orchestration system...",  # summary
        "",  # referenced documents
    )
    llm_request = LLMRequest(query="Tell me about Kubernetes")
    response = ols.conversation_request(llm_request)
    assert (
        response.response
        == "Kubernetes is an open-source container-orchestration system..."
    )
    assert len(response.conversation_id) > 0

    # valid question, yaml
    mock_validate_question.return_value = [
        constants.SUBJECT_VALID,
        constants.CATEGORY_YAML,
    ]
    mock_generate_yaml.return_value = "content: generated yaml"
    llm_request = LLMRequest(query="Generate a yaml")
    response = ols.conversation_request(llm_request)
    assert response.response == "content: generated yaml"
    assert len(response.conversation_id) > 0

    # valid question, yaml, generator failure
    mock_validate_question.return_value = [
        constants.SUBJECT_VALID,
        constants.CATEGORY_YAML,
    ]
    mock_generate_yaml.side_effect = Exception
    with pytest.raises(HTTPException) as excinfo:
        llm_request = LLMRequest(query="Generate a yaml")
        response = ols.conversation_request(llm_request)
        assert excinfo.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        assert len(response.conversation_id) == 0

    # question of unknown type
    mock_validate_question.return_value = [
        constants.SUBJECT_VALID,
        constants.CATEGORY_UNKNOWN,
    ]
    llm_request = LLMRequest(query="ask a question of unknown type")
    response = ols.conversation_request(llm_request)
    assert response.response == str(
        {
            "detail": {
                "response": "Question does not provide enough context, \
                Please rephrase your question or provide more detail"
            }
        }
    )
    assert len(response.conversation_id) > 0

    # invalid question
    mock_validate_question.return_value = [
        constants.SUBJECT_INVALID,
        constants.CATEGORY_YAML,
    ]
    llm_request = LLMRequest(query="Generate a yaml")
    response = ols.conversation_request(llm_request)
    assert response.response == str(
        {
            "detail": {
                "response": "I can only answer questions about \
            OpenShift and Kubernetes. Please rephrase your question"
            }
        }
    )
    assert len(response.conversation_id) > 0

    # conversation is cached
    mock_validate_question.return_value = [
        constants.SUBJECT_VALID,
        constants.CATEGORY_YAML,
    ]
    mock_generate_yaml.return_value = "content: generated yaml"
    mock_generate_yaml.side_effect = None
    mock_conversation_cache_get.return_value = "previous conversation input"
    llm_request = LLMRequest(query="Generate a yaml", conversation_id=Utils.get_suid())
    response = ols.conversation_request(llm_request)
    assert response.response == "content: generated yaml"

    # validation failure
    mock_validate_question.side_effect = HTTPException
    with pytest.raises(HTTPException) as excinfo:
        llm_request = LLMRequest(query="Generate a yaml")
        response = ols.conversation_request(llm_request)
        assert excinfo.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        assert len(response.conversation_id) == 0


def fake_llm_chain_call(self, **kwargs):
    """Fake llm chain call."""
    inputs = kwargs.get("inputs", {})
    query = inputs.get("query", "")
    return f"response to: {query}"


@patch("ols.app.endpoints.ols.LLMChain", new=mock_llm_chain({"text": "llm response"}))
@patch("ols.app.endpoints.ols.LLMLoader", new=mock_llm_loader(None))
def test_conversation_request_debug_api(load_config):
    """Test conversation request debug API endpoint."""
    # no conversation id
    llm_request = LLMRequest(query="Tell me about Kubernetes")
    response = ols.conversation_request_debug_api(llm_request)
    assert response.response == "llm response"

    # with conversation id
    llm_request = LLMRequest(query="Tell me about Kubernetes", conversation_id="123")
    response = ols.conversation_request_debug_api(llm_request)
    assert response.response == "llm response"
