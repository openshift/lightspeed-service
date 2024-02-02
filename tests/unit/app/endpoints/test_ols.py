"""Unit tests for OLS endpoint."""

from http import HTTPStatus
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from ols import constants
from ols.app.endpoints import ols
from ols.app.models.models import LLMRequest
from ols.utils import config, suid
from tests.mock_classes.llm_chain import mock_llm_chain
from tests.mock_classes.llm_loader import mock_llm_loader


@pytest.fixture(scope="module")
def load_config():
    """Load config before unit tests."""
    config.init_config("tests/config/test_app_endpoints.yaml")


@patch("ols.src.query_helpers.question_validator.QuestionValidator.validate_question")
@patch("ols.src.query_helpers.docs_summarizer.DocsSummarizer.summarize")
@patch("ols.utils.config.conversation_cache.get")
def test_conversation_request(
    mock_conversation_cache_get,
    mock_summarize,
    mock_validate_question,
    load_config,
):
    """Test conversation request API endpoint."""
    # valid question, no yaml
    mock_validate_question.return_value = constants.SUBJECT_VALID
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
    mock_validate_question.return_value = constants.SUBJECT_VALID
    mock_summarize.return_value = ("content: generated yaml", "")
    llm_request = LLMRequest(query="Generate a yaml")
    response = ols.conversation_request(llm_request)
    assert response.response == "content: generated yaml"
    assert len(response.conversation_id) > 0

    # valid question, yaml, generator failure
    mock_validate_question.return_value = constants.SUBJECT_VALID
    mock_summarize.side_effect = Exception
    with pytest.raises(HTTPException) as excinfo:
        llm_request = LLMRequest(query="Generate a yaml")
        response = ols.conversation_request(llm_request)
        assert excinfo.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        assert len(response.conversation_id) == 0

    # invalid question
    mock_validate_question.return_value = constants.SUBJECT_INVALID
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
    mock_validate_question.return_value = constants.SUBJECT_VALID
    mock_summarize.return_value = ("content: generated yaml", "")
    mock_summarize.side_effect = None
    mock_conversation_cache_get.return_value = "previous conversation input"
    llm_request = LLMRequest(query="Generate a yaml", conversation_id=suid.get_suid())
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
