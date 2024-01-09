import requests
from fastapi.testclient import TestClient

from app import constants
from app.endpoints import ols
from app.main import app
from src.query_helpers.question_validator import QuestionValidator

client = TestClient(app)


def test_healthz() -> None:
    response = client.get("/healthz")
    assert response.status_code == requests.codes.ok
    assert response.json() == {"status": "1"}


def test_readyz() -> None:
    response = client.get("/readyz")
    assert response.status_code == requests.codes.ok
    assert response.json() == {"status": "1"}


def test_root() -> None:
    response = client.get("/")
    assert response.status_code == requests.codes.ok
    assert response.json() == {
        "message": "This is the default endpoint for OLS",
        "status": "running",
    }


def test_status() -> None:
    response = client.get("/status")
    assert response.status_code == requests.codes.ok
    assert response.json() == {
        "message": "This is the default endpoint for OLS",
        "status": "running",
    }


def test_raw_prompt(monkeypatch) -> None:
    # the raw prompt should just return stuff from LangChainInterface, so mock that base method
    # model_context is what imports LangChainInterface, so we have to mock that particular usage/"instance"
    # of it in our tests

    from tests.mock_classes.langchain_interface import mock_langchain_interface
    from tests.mock_classes.llm_loader import mock_llm_loader

    ml = mock_langchain_interface("test response")

    monkeypatch.setattr(ols, "LLMLoader", mock_llm_loader(ml()))

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
