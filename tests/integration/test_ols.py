"""Integration tests for basic OLS REST API endpoints."""

import os
from unittest.mock import patch

import requests
from fastapi.testclient import TestClient

from ols import constants
from ols.app.models.config import ProviderConfig
from ols.app.utils import Utils
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
# during LLM init, exceptins will occur on CI, so let's mock that too
@patch("ols.app.endpoints.ols.LLMLoader", new=mock_llm_loader(None))
def test_debug_query() -> None:
    """Check the REST API /v1/debug/query with POST HTTP method when expected payload is posted."""
    conversation_id = Utils.get_suid()
    response = client.post(
        "/v1/debug/query",
        json={"conversation_id": conversation_id, "query": "test query"},
    )
    print(response)
    assert response.status_code == requests.codes.ok
    assert response.json() == {
        "conversation_id": conversation_id,
        "query": "test query",
        "response": "test response",
        "provider": None,  # default value in request
        "model": None,  # default value in request
    }


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
