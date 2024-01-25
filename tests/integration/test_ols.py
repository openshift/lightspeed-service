"""Integration tests for basic OLS REST API endpoints."""

import os
from unittest.mock import patch

import requests
from fastapi.testclient import TestClient

from ols import constants
from tests.mock_classes.llm_chain import mock_llm_chain
from tests.mock_classes.llm_loader import mock_llm_loader

# config file path needs to be set before importing app which tries to load the config
os.environ["OLS_CONFIG_FILE"] = "tests/config/valid_config.yaml"
from ols.app.main import app  # noqa: E402

client = TestClient(app)


def test_liveness() -> None:
    """Test handler for /liveness REST API endpoint."""
    response = client.get("/liveness")
    assert response.status_code == requests.codes.ok
    assert response.json() == {"status": "1"}


def test_readiness() -> None:
    """Test handler for /readiness REST API endpoint."""
    response = client.get("/readiness")
    assert response.status_code == requests.codes.ok
    assert response.json() == {"status": "1"}


# the raw prompt should just return stuff from LLMChain, so mock that base method in ols.py
@patch("ols.app.endpoints.ols.LLMChain", new=mock_llm_chain({"text": "test response"}))
# during LLM init, exceptins will occur on CI, so let's mock that too
@patch("ols.app.endpoints.ols.LLMLoader", new=mock_llm_loader(None))
def test_debug_query() -> None:
    """Check the REST API /v1/debug/query with POST HTTP method when expected payload is posted."""
    response = client.post(
        "/v1/debug/query", json={"conversation_id": "1234", "query": "test query"}
    )
    print(response)
    assert response.status_code == requests.codes.ok
    assert response.json() == {
        "conversation_id": "1234",
        "query": "test query",
        "response": "test response",
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
                "url": "https://errors.pydantic.dev/2.5/v/model_attributes_type",
            }
        ],
    }


def test_post_question_on_invalid_question() -> None:
    """Check the REST API /v1/query with POST HTTP method for invalid question."""
    # let's pretend the question is invalid without even asking LLM
    answer = (constants.INVALID, "anything")
    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question", return_value=answer
    ):
        response = client.post(
            "/v1/query", json={"conversation_id": "1234", "query": "test query"}
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
            "conversation_id": "1234",
            "query": "test query",
            "response": expected_details,
        }
        assert response.json() == expected_json


def test_post_question_on_unknown_response_type() -> None:
    """Check the REST API /v1/query with POST HTTP method when unknown response type is returned."""
    # let's pretend the question is valid, but there's an error, without even asking LLM
    answer = (constants.VALID, constants.REPHRASE)
    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question", return_value=answer
    ):
        response = client.post(
            "/v1/query", json={"conversation_id": "1234", "query": "test query"}
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
            "conversation_id": "1234",
            "query": "test query",
            "response": expected_details,
        }
        assert response.json() == expected_json
