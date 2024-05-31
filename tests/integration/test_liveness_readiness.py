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
    # index is not loaded, but there is reference content in config
    # - service should not be ready
    config._rag_index = None
    assert config.ols_config.reference_content is not None
    response = client.get("/readiness")
    assert response.status_code == requests.codes.ok
    assert response.json() == {"ready": False, "reason": "index is not ready"}

    # index is not loaded, but it shouldn't as there is no reference
    # content in config - service should be ready
    config.ols_config.reference_content = None
    config._rag_index = None
    response = client.get("/readiness")
    assert response.status_code == requests.codes.ok
    assert response.json() == {"ready": True, "reason": "service is ready"}

    # index is loaded - service should be ready
    config._rag_index = "something else than None"
    response = client.get("/readiness")
    assert response.status_code == requests.codes.ok
    assert response.json() == {"ready": True, "reason": "service is ready"}
