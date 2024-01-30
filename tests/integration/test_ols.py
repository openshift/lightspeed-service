"""Integration tests for basic OLS REST API endpoints."""

import os
from unittest.mock import patch

import requests
from fastapi.testclient import TestClient

from ols import constants
from ols.app.models.config import ProviderConfig
from ols.app.utils import Utils
from ols.src.llms.llm_loader import LLMConfigurationError
from tests.mock_classes.llm_chain import mock_llm_chain
from tests.mock_classes.llm_loader import mock_llm_loader


# we need to patch the config file path to point to the test
# config file before we import anything from main.py
@patch.dict(os.environ, {"OLS_CONFIG_FILE": "tests/config/valid_config.yaml"})
def setup():
    """Setups the test client."""
    global client
    from ols.app.main import app

    client = TestClient(app)


# the raw prompt should just return stuff from LLMChain, so mock that base method in ols.py
@patch("ols.app.endpoints.ols.LLMChain", new=mock_llm_chain({"text": "test response"}))
# during LLM init, exceptions will occur on CI, so let's mock that too
@patch("ols.app.endpoints.ols.LLMLoader", new=mock_llm_loader(None))
def test_debug_query() -> None:
    """Check the REST API /v1/debug/query with POST HTTP method when expected payload is posted."""
    conversation_id = Utils.get_suid()
    response = client.post(
        "/v1/debug/query",
        json={"conversation_id": conversation_id, "query": "test query"},
    )

    assert response.status_code == requests.codes.ok
    assert response.json() == {
        "conversation_id": conversation_id,
        "query": "test query",
        "response": "test response",
        "provider": None,  # default value in request
        "model": None,  # default value in request
    }


# the raw prompt should just return stuff from LLMChain, so mock that base method in ols.py
@patch("ols.app.endpoints.ols.LLMChain", new=mock_llm_chain({"text": "test response"}))
# during LLM init, exceptions will occur on CI, so let's mock that too
@patch("ols.app.endpoints.ols.LLMLoader", new=mock_llm_loader(None))
def test_debug_query_no_conversation_id() -> None:
    """Check the REST API /v1/debug/query with POST HTTP method conversation ID is not provided."""
    response = client.post(
        "/v1/debug/query",
        json={"query": "test query"},
    )

    assert response.status_code == requests.codes.ok
    json = response.json()

    # check that conversation ID is being created
    assert len(json["conversation_id"]) > 0
    assert json["query"] == "test query"
    assert json["response"] == "test response"
    assert json["provider"] is None  # default value in request
    assert json["model"] is None  # default value in request


# the raw prompt should just return stuff from LLMChain, so mock that base method in ols.py
@patch("ols.app.endpoints.ols.LLMChain", new=mock_llm_chain({"text": "test response"}))
# during LLM init, exceptions will occur on CI, so let's mock that too
@patch("ols.app.endpoints.ols.LLMLoader", new=mock_llm_loader(None))
def test_debug_query_no_query() -> None:
    """Check the REST API /v1/debug/query with POST HTTP method when query is not specified."""
    conversation_id = Utils.get_suid()
    response = client.post(
        "/v1/debug/query",
        json={"conversation_id": conversation_id},
    )

    # request can't be processed correctly
    assert response.status_code == requests.codes.unprocessable


# the raw prompt should just return stuff from LLMChain, so mock that base method in ols.py
@patch("ols.app.endpoints.ols.LLMChain", new=mock_llm_chain({"text": "test response"}))
# during LLM init, exceptions will occur on CI, so let's mock that too
@patch("ols.app.endpoints.ols.LLMLoader", new=mock_llm_loader(None))
def test_debug_query_no_payload() -> None:
    """Check the REST API /v1/debug/query with POST HTTP method when payload is empty."""
    response = client.post("/v1/debug/query")

    # request can't be processed correctly
    assert response.status_code == requests.codes.unprocessable


def test_post_question_on_unexpected_payload() -> None:
    """Check the REST API /v1/query with POST HTTP method when unexpected payload is posted."""
    response = client.post("/v1/query", json="this is really not proper payload")
    assert response.status_code == requests.codes.unprocessable
    assert response.json() == {
        "detail": [
            {
                "input": "this is really not proper payload",
                "loc": ["body"],
                "msg": "Input should be a valid dictionary or object to extract fields from",
                "type": "model_attributes_type",
                "url": "https://errors.pydantic.dev/2.6/v/model_attributes_type",
            }
        ],
    }


def test_post_question_without_payload() -> None:
    """Check the REST API /v1/query with POST HTTP method when no payload is posted."""
    # perform POST request without any payload
    response = client.post("/v1/query")
    assert response.status_code == requests.codes.unprocessable

    # check the response payload
    json = response.json()
    assert "detail" in json, "Missing 'detail' node in response payload"
    detail = json["detail"][0]
    assert detail["input"] is None
    assert "Field required" in detail["msg"]


def test_post_question_on_invalid_question() -> None:
    """Check the REST API /v1/query with POST HTTP method for invalid question."""
    # let's pretend the question is invalid without even asking LLM
    answer = (constants.SUBJECT_INVALID, "anything")
    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question", return_value=answer
    ):
        conversation_id = Utils.get_suid()
        response = client.post(
            "/v1/query",
            json={"conversation_id": conversation_id, "query": "test query"},
        )
        assert response.status_code == requests.codes.ok
        expected_details = str(
            {
                "detail": {
                    "response": "I can only answer questions about \
            OpenShift and Kubernetes. Please rephrase your question"
                }
            }
        )
        expected_json = {
            "conversation_id": conversation_id,
            "query": "test query",
            "response": expected_details,
            "provider": None,  # default value in request
            "model": None,  # default value in request
        }
        assert response.json() == expected_json


def test_post_question_on_generic_response_type_summarize_error() -> None:
    """Check the REST API /v1/query with POST HTTP method when generic response type is returned."""
    # let's pretend the question is valid and generic one
    answer = (constants.SUBJECT_VALID, constants.CATEGORY_GENERIC)
    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question", return_value=answer
    ):
        with patch(
            "ols.app.endpoints.ols.DocsSummarizer.summarize",
            side_effect=Exception("summarizer error"),
        ):
            conversation_id = Utils.get_suid()
            response = client.post(
                "/v1/query",
                json={"conversation_id": conversation_id, "query": "test query"},
            )
            assert response.status_code == requests.codes.internal_server_error
            expected_json = {"detail": "Error while obtaining answer for user question"}
            assert response.json() == expected_json


def test_post_question_on_generic_response_llm_configuration_error() -> None:
    """Check the REST API /v1/query with POST HTTP method when generic response type is returned."""
    # let's pretend the question is valid and generic one
    answer = (constants.SUBJECT_VALID, constants.CATEGORY_GENERIC)
    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question", return_value=answer
    ):
        with patch(
            "ols.app.endpoints.ols.DocsSummarizer.summarize",
            side_effect=LLMConfigurationError("LLM configuration error"),
        ):
            conversation_id = Utils.get_suid()
            response = client.post(
                "/v1/query",
                json={"conversation_id": conversation_id, "query": "test query"},
            )
            assert response.status_code == requests.codes.unprocessable
            expected_json = {
                "detail": {
                    "response": "Unable to process this request because 'LLM configuration error'"
                }
            }
            assert response.json() == expected_json


def test_post_question_on_unknown_response_type() -> None:
    """Check the REST API /v1/query with POST HTTP method when unknown response type is returned."""
    # let's pretend the question is valid, but there's an error, without even asking LLM
    answer = (constants.SUBJECT_VALID, constants.CATEGORY_UNKNOWN)
    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question", return_value=answer
    ):
        conversation_id = Utils.get_suid()
        response = client.post(
            "/v1/query",
            json={"conversation_id": conversation_id, "query": "test query"},
        )
        assert response.status_code == requests.codes.ok
        expected_details = str(
            {
                "detail": {
                    "response": "Question does not provide enough context, \
                Please rephrase your question or provide more detail"
                }
            }
        )
        expected_json = {
            "conversation_id": conversation_id,
            "query": "test query",
            "response": expected_details,
            "provider": None,  # default value in request
            "model": None,  # default value in request
        }
        assert response.json() == expected_json


def test_post_question_that_is_not_validated() -> None:
    """Check the REST API /v1/query with POST HTTP method for question that is not validated."""
    # let's pretend the question can not be validated
    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question",
        side_effect=Exception("can not validate"),
    ):
        conversation_id = Utils.get_suid()
        response = client.post(
            "/v1/query",
            json={"conversation_id": conversation_id, "query": "test query"},
        )

        # error should be returned
        assert response.status_code == requests.codes.internal_server_error
        expected_details = {"detail": "Error while validating question"}
        assert response.json() == expected_details


def test_post_question_with_provider_but_not_model() -> None:
    """Check how missing model is detected in request."""
    conversation_id = Utils.get_suid()
    response = client.post(
        "/v1/query",
        json={
            "conversation_id": conversation_id,
            "query": "test query",
            "provider": constants.PROVIDER_BAM,
        },
    )
    assert response.status_code == requests.codes.unprocessable
    expected_json = {
        "detail": {"response": "LLM model must be specified when provider is specified"}
    }
    assert response.json() == expected_json


def test_post_question_with_model_but_not_provider() -> None:
    """Check how missing provider is detected in request."""
    conversation_id = Utils.get_suid()
    response = client.post(
        "/v1/query",
        json={
            "conversation_id": conversation_id,
            "query": "test query",
            "model": constants.GRANITE_13B_CHAT_V1,
        },
    )
    assert response.status_code == requests.codes.unprocessable
    expected_json = {
        "detail": {
            "response": "LLM provider must be specified when the model is specified"
        }
    }
    assert response.json() == expected_json


class TestQuery:
    """Test the /v1/query endpoint."""

    def test_unsupported_provider_in_post(self):
        """Check the REST API /v1/query with POST method when unsupported provider is requested."""
        # empty config - no providers
        with patch("ols.utils.config.llm_config.providers", new={}):
            response = client.post(
                "/v1/query",
                json={
                    "query": "hello?",
                    "provider": "some-provider",
                    "model": "some-model",
                },
            )

            assert response.status_code == requests.codes.unprocessable
            assert response.json() == {
                "detail": {
                    "response": "Unable to process this request because "
                    "'Unsupported LLM provider some-provider'"
                }
            }

    def test_unsupported_model_in_post(self):
        """Check the REST API /v1/query with POST method when unsupported model is requested."""
        test_provider = "test-provider"
        provider_config = ProviderConfig()
        provider_config.models = {}  # no models configured

        with patch(
            "ols.utils.config.llm_config.providers",
            new={test_provider: provider_config},
        ):
            response = client.post(
                "/v1/query",
                json={
                    "query": "hello?",
                    "provider": test_provider,
                    "model": "some-model",
                },
            )

            assert response.status_code == requests.codes.unprocessable
            assert response.json() == {
                "detail": {
                    "response": "Unable to process this request because "
                    "'No configuration provided for model some-model under "
                    "LLM provider test-provider'"
                }
            }
