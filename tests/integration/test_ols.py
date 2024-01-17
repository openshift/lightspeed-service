"""Integration tests for basic OLS REST API endpoints."""

from unittest.mock import patch

import requests
from fastapi.testclient import TestClient

from ols import constants
from ols.app.main import app
from ols.src.query_helpers.question_validator import QuestionValidator
from tests.mock_classes.llm_chain import mock_llm_chain

client = TestClient(app)


def test_healthz() -> None:
    """Test handler for /healthz REST API endpoint."""
    response = client.get("/healthz")
    assert response.status_code == requests.codes.ok
    assert response.json() == {"status": "1"}


def test_readyz() -> None:
    """Test handler for /readyz REST API endpoint."""
    response = client.get("/readyz")
    assert response.status_code == requests.codes.ok
    assert response.json() == {"status": "1"}


def test_root() -> None:
    """Test handler for / REST API endpoint."""
    response = client.get("/")
    assert response.status_code == requests.codes.ok
    assert response.json() == {
        "message": "This is the default endpoint for OLS",
        "status": "running",
    }


def test_status() -> None:
    """Test handler for /status REST API endpoint."""
    response = client.get("/status")
    assert response.status_code == requests.codes.ok
    assert response.json() == {
        "message": "This is the default endpoint for OLS",
        "status": "running",
    }


# the raw prompt should just return stuff from LLMChain, so mock that base method in ols.py
@patch("ols.app.endpoints.ols.LLMChain", new=mock_llm_chain("test response"))
def test_raw_prompt(monkeypatch) -> None:
    """Check the REST API /ols/raw_prompt with POST HTTP method when expected payload is posted."""
    response = client.post(
        "/ols/raw_prompt", json={"conversation_id": "1234", "query": "test query"}
    )
    print(response)
    assert response.status_code == requests.codes.ok
    assert response.json() == {
        "conversation_id": "1234",
        "query": "test query",
        "response": "test response",
    }


def test_post_question_on_unexpected_payload() -> None:
    """Check the REST API /ols/ with POST HTTP method when unexpected payload is posted."""
    response = client.post("/ols/", json="this is really not proper payload")
    print(response)
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


def test_post_question_on_invalid_question(monkeypatch) -> None:
    """Check the REST API /ols/ with POST HTTP method for invalid question."""

    def dummy_validator(
        self, conversation: str, query: str, verbose: bool = False
    ) -> list[str]:
        return constants.INVALID, "anything"

    # let's pretend the question is invalid without even asking LLM
    monkeypatch.setattr(QuestionValidator, "validate_question", dummy_validator)

    response = client.post(
        "/ols/", json={"conversation_id": "1234", "query": "test query"}
    )
    print(response)
    assert response.status_code == requests.codes.unprocessable
    assert response.json() == {
        "detail": {
            "response": "Sorry, I can only answer questions about OpenShift "
            "and Kubernetes. This does not look like something I "
            "know how to handle."
        }
    }


def test_post_question_on_unknown_response_type(monkeypatch) -> None:
    """Check the REST API /ols/ with POST HTTP method when unknown response type is returned."""

    def dummy_validator(
        self, conversation: str, query: str, verbose: bool = False
    ) -> list[str]:
        return constants.VALID, constants.SOME_FAILURE

    # let's pretend the question is valid without even asking LLM
    # but the question type is unknown
    monkeypatch.setattr(QuestionValidator, "validate_question", dummy_validator)

    response = client.post(
        "/ols/", json={"conversation_id": "1234", "query": "test query"}
    )
    print(response)
    assert response.status_code == requests.codes.internal_server_error
    assert response.json() == {
        "detail": {"response": "Internal server error. Please try again."}
    }
