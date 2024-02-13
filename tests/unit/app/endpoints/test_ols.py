"""Unit tests for OLS endpoint."""

from http import HTTPStatus
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException

from ols import constants
from ols.app.endpoints import ols
from ols.app.models.models import LLMRequest
from ols.src.llms.llm_loader import LLMConfigurationError
from ols.utils import config, suid
from tests.mock_classes.llm_chain import mock_llm_chain
from tests.mock_classes.llm_loader import mock_llm_loader


@pytest.fixture(scope="module")
def load_config():
    """Load config before unit tests."""
    config.init_config("tests/config/test_app_endpoints.yaml")


def test_retrieve_conversation_new_id(load_config):
    """Check the function to retrieve conversation ID."""
    llm_request = LLMRequest(query="Tell me about Kubernetes", conversation_id=None)
    new_id = ols.retrieve_conversation_id(llm_request)
    assert suid.check_suid(new_id), "Improper conversation ID generated"


def test_retrieve_conversation_id_existing_id(load_config):
    """Check the function to retrieve conversation ID when one already exists."""
    old_id = suid.get_suid()
    llm_request = LLMRequest(query="Tell me about Kubernetes", conversation_id=old_id)
    new_id = ols.retrieve_conversation_id(llm_request)
    assert new_id == old_id, "Old (existing) ID should be retrieved." ""


# TODO: distribute individual test cases to separate test functions
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
    # valid question
    mock_validate_question.return_value = constants.SUBJECT_VALID
    mock_response = Mock()
    mock_response.response = (
        "Kubernetes is an open-source container-orchestration system..."  # summary
    )
    mock_summarize.return_value = (
        mock_response,
        "",  # referenced documents
    )
    llm_request = LLMRequest(query="Tell me about Kubernetes")
    response = ols.conversation_request(llm_request)
    assert (
        response.response
        == "Kubernetes is an open-source container-orchestration system..."
    )
    assert len(response.conversation_id) > 0

    # invalid question
    mock_validate_question.return_value = constants.SUBJECT_INVALID
    llm_request = LLMRequest(query="Generate a yaml")
    response = ols.conversation_request(llm_request)
    assert response.response == (
        "I can only answer questions about OpenShift and Kubernetes. "
        "Please rephrase your question"
    )
    assert len(response.conversation_id) > 0

    # validation failure
    mock_validate_question.side_effect = HTTPException
    with pytest.raises(HTTPException) as excinfo:
        llm_request = LLMRequest(query="Generate a yaml")
        response = ols.conversation_request(llm_request)
        assert excinfo.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        assert len(response.conversation_id) == 0


@patch("ols.src.query_helpers.question_validator.QuestionValidator.validate_question")
@patch("ols.utils.config.conversation_cache.get")
def test_conversation_request_on_wrong_configuration(
    mock_conversation_cache_get,
    mock_validate_question,
    load_config,
):
    """Test conversation request API endpoint."""
    # mock invalid configuration
    message = "wrong model is configured"
    mock_validate_question.side_effect = Mock(
        side_effect=LLMConfigurationError(message)
    )
    llm_request = LLMRequest(query="Tell me about Kubernetes")

    # call must fail because we mocked invalid configuration state
    with pytest.raises(
        HTTPException, match=f"Unable to process this request because '{message}'"
    ):
        ols.conversation_request(llm_request)


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
