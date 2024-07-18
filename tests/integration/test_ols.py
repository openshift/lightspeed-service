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
