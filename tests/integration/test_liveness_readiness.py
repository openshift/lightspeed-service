"""Integration tests for /livenss and /readiness REST API endpoints."""

import os
from unittest.mock import patch

import pytest
import requests
from fastapi.testclient import TestClient

from ols.utils import config


# we need to patch the config file path to point to the test
# config file before we import anything from main.py
@pytest.fixture(scope="module")
@patch.dict(os.environ, {"OLS_CONFIG_FILE": "tests/config/valid_config.yaml"})
def setup():
    """Setups the test client."""
    global client
    from ols.app.main import app

    config.init_config("tests/config/valid_config.yaml")
    client = TestClient(app)


def test_liveness(setup):
    """Test handler for /liveness REST API endpoint."""
    response = client.get("/liveness")
    assert response.status_code == requests.codes.ok
    assert response.json() == {"status": {"status": "healthy"}}


def test_readiness(setup):
    """Test handler for /readiness REST API endpoint."""
    response = client.get("/readiness")
    assert response.status_code == requests.codes.ok
    assert response.json() == {"status": {"status": "healthy"}}


def test_liveness_head_http_method(setup) -> None:
    """Test handler for /liveness REST API endpoint when HEAD HTTP method is used."""
    response = client.head("/liveness")
    assert response.status_code == requests.codes.ok
    assert response.text == ""


def test_readiness_head_http_method(setup) -> None:
    """Test handler for /readiness REST API endpoint when HEAD HTTP method is used."""
    response = client.head("/readiness")
    assert response.status_code == requests.codes.ok
    assert response.text == ""
