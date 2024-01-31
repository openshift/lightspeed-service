"""Integration tests for REST API endpoint that provides OpenAPI specification."""

import os
from unittest.mock import patch

import requests
from fastapi.testclient import TestClient


# we need to patch the config file path to point to the test
# config file before we import anything from main.py
@patch.dict(os.environ, {"OLS_CONFIG_FILE": "tests/config/valid_config.yaml"})
def setup():
    """Setups the test client."""
    global client
    from ols.app.main import app

    client = TestClient(app)


def test_openapi_endpoint():
    """Check if REST API provides endpoint with OpenAPI specification."""
    response = client.get("/openapi.json")
    assert response.status_code == requests.codes.ok

    # this line ensures that response payload contains proper JSON
    payload = response.json()
    assert payload is not None, "Incorrect response"

    # check the metadata nodes
    for attribute in ("openapi", "info", "components", "paths"):
        assert (
            attribute in payload
        ), f"Required metadata attribute {attribute} not found"

    # check application description
    info = payload["info"]
    assert "description" in info, "Service description not provided"
    assert "OpenShift LightSpeed Service API specification" in info["description"]

    # elementary check that all mandatory endpoints are covered
    paths = payload["paths"]
    for endpoint in ("/readiness", "/liveness", "/v1/query", "/v1/feedback"):
        assert endpoint in paths, f"Endpoint {endpoint} is not described"


def test_openapi_endpoint_head_method():
    """Check if REST API allows HEAD HTTP method for endpoint with OpenAPI specification."""
    response = client.head("/openapi.json")
    assert response.status_code == requests.codes.ok
    assert response.text == ""
