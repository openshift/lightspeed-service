"""Integration tests for /livenss and /readiness REST API endpoints."""

import os
from unittest.mock import patch

import pytest
import requests
from fastapi.testclient import TestClient

from ols import config


# we need to patch the config file path to point to the test
# config file before we import anything from main.py
@pytest.fixture(scope="function")
@patch.dict(os.environ, {"OLS_CONFIG_FILE": "tests/config/valid_config.yaml"})
def _setup():
    """Setups the test client."""
    global client
    config.reload_from_yaml_file("tests/config/valid_config.yaml")

    # app.main need to be imported after the configuration is read
    from ols.app.main import app

    client = TestClient(app)


def test_liveness(_setup):
    """Test handler for /liveness REST API endpoint."""
    response = client.get("/liveness")
    assert response.status_code == requests.codes.ok
    assert response.json() == {"alive": True}


def test_readiness(_setup):
    """Test handler for /readiness REST API endpoint."""
    # index is not ready
    with (
        patch("ols.app.endpoints.health.llm_is_ready", return_value=True),
        patch("ols.app.endpoints.health.index_is_ready", return_value=False),
    ):
        response = client.get("/readiness")
        assert response.status_code == requests.codes.ok
        assert response.json() == {"ready": False, "reason": "index is not ready"}

    # llm is not ready
    with (
        patch("ols.app.endpoints.health.llm_is_ready", return_value=False),
        patch("ols.app.endpoints.health.index_is_ready", return_value=True),
    ):
        response = client.get("/readiness")
        assert response.status_code == requests.codes.ok
        assert response.json() == {"ready": False, "reason": "LLM is not ready"}

    # everything is ready
    with (
        patch("ols.app.endpoints.health.llm_is_ready", return_value=True),
        patch("ols.app.endpoints.health.index_is_ready", return_value=True),
    ):
        response = client.get("/readiness")
        assert response.status_code == requests.codes.ok
        assert response.json() == {"ready": True, "reason": "service is ready"}
