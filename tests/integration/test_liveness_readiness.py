"""Integration tests for /livenss and /readiness REST API endpoints."""

import os
from unittest.mock import patch

import pytest
import requests
from fastapi.testclient import TestClient

from ols import config
from ols.constants import CONFIGURATION_FILE_NAME_ENV_VARIABLE

client: TestClient


# we need to patch the config file path to point to the test
# config file before we import anything from main.py
@pytest.fixture(scope="function", autouse=True)
@patch.dict(
    os.environ,
    {
        CONFIGURATION_FILE_NAME_ENV_VARIABLE: "tests/config/config_for_integration_tests.yaml"
    },
)
def _setup():
    """Setups the test client."""
    global client
    config.reload_from_yaml_file("tests/config/config_for_integration_tests.yaml")

    # app.main need to be imported after the configuration is read
    from ols.app.main import app  # pylint: disable=C0415

    client = TestClient(app)


def test_liveness():
    """Test handler for /liveness REST API endpoint."""
    response = client.get("/liveness")
    assert response.status_code == requests.codes.ok
    assert response.json() == {"alive": True}


def test_readiness():
    """Test handler for /readiness REST API endpoint."""
    # index is not ready
    with (
        patch("ols.app.endpoints.health.llm_is_ready", return_value=True),
        patch("ols.app.endpoints.health.index_is_ready", return_value=False),
    ):
        response = client.get("/readiness")
        assert response.status_code == requests.codes.service_unavailable
        assert response.json() == {
            "detail": {
                "response": "Service is not ready",
                "cause": "Index is not ready",
            }
        }

    # index is ready, LLM is not ready
    with (
        patch("ols.app.endpoints.health.llm_is_ready", return_value=False),
        patch("ols.app.endpoints.health.index_is_ready", return_value=True),
    ):
        response = client.get("/readiness")
        assert response.status_code == requests.codes.service_unavailable
        assert response.json() == {
            "detail": {"response": "Service is not ready", "cause": "LLM is not ready"}
        }

    # both index and LLM are ready
    with (
        patch("ols.app.endpoints.health.llm_is_ready", return_value=True),
        patch("ols.app.endpoints.health.index_is_ready", return_value=True),
    ):
        response = client.get("/readiness")
        assert response.status_code == requests.codes.ok
        assert response.json() == {"ready": True, "reason": "service is ready"}
