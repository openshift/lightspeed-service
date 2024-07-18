"""Integration tests for basic OLS REST API endpoints."""

from unittest.mock import patch

import pytest
import requests
from fastapi.testclient import TestClient

from ols import config, constants
from ols.app.models.config import (
    ProviderConfig,
    QueryFilter,
)
from ols.utils import suid
from tests.mock_classes.mock_llm_chain import mock_llm_chain
from tests.mock_classes.mock_llm_loader import mock_llm_loader


@pytest.fixture(scope="function")
def _setup():
    """Setups the test client."""
    config.reload_from_yaml_file("tests/config/config_for_integration_tests.yaml")
    global client

    # app.main need to be imported after the configuration is read
    from ols.app.main import app

    client = TestClient(app)


def test_post_question_on_unexpected_payload(_setup):
    """Check the REST API /v1/query with POST HTTP method when unexpected payload is posted."""
    response = client.post("/v1/query", json="this is really not proper payload")
    assert response.status_code == requests.codes.unprocessable

    # try to deserialize payload
    response_json = response.json()

    # remove attribute that strongly depends on Pydantic version
    if "url" in response_json["detail"][0]:
        del response_json["detail"][0]["url"]

    assert response_json == {
        "detail": [
            {
                "input": "this is really not proper payload",
                "loc": ["body"],
                "msg": "Input should be a valid dictionary or object to extract fields from",
                "type": "model_attributes_type",
            }
        ],
    }


def test_post_question_without_payload(_setup):
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


def test_post_question_on_invalid_question(_setup):
    """Check the REST API /v1/query with POST HTTP method for invalid question."""
    # let's pretend the question is invalid without even asking LLM
    with patch(
        "ols.app.endpoints.ols.QuestionValidator.validate_question", return_value=False
    ):
        conversation_id = suid.get_suid()
        response = client.post(
            "/v1/query",
            json={"conversation_id": conversation_id, "query": "test query"},
        )
        assert response.status_code == requests.codes.ok

        expected_json = {
            "conversation_id": conversation_id,
            "response": constants.INVALID_QUERY_RESP,
            "referenced_documents": [],
            "truncated": False,
        }
        assert response.json() == expected_json


def test_post_question_on_generic_response_type_summarize_error(_setup):
    """Check the REST API /v1/query with POST HTTP method when generic response type is returned."""
    # let's pretend the question is valid and generic one
    answer = constants.SUBJECT_ALLOWED
    with (
        patch(
            "ols.app.endpoints.ols.QuestionValidator.validate_question",
            return_value=answer,
        ),
        patch(
            "ols.app.endpoints.ols.DocsSummarizer.summarize",
            side_effect=Exception("summarizer error"),
        ),
    ):
        conversation_id = suid.get_suid()
        response = client.post(
            "/v1/query",
            json={"conversation_id": conversation_id, "query": "test query"},
        )
        assert response.status_code == requests.codes.internal_server_error
        expected_json = {
            "detail": {
                "cause": "summarizer error",
                "response": "Error while obtaining answer for user question",
            }
        }

        assert response.json() == expected_json
