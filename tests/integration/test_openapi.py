"""Integration tests for REST API endpoint that provides OpenAPI specification."""

import json
import os
from unittest.mock import patch

import pytest
import requests
from fastapi.testclient import TestClient


# we need to patch the config file path to point to the test
# config file before we import anything from main.py
@pytest.fixture(scope="module")
@patch.dict(os.environ, {"OLS_CONFIG_FILE": "tests/config/valid_config.yaml"})
def setup():
    """Setups the test client."""
    global client
    from ols.app.main import app

    client = TestClient(app)


def test_openapi_endpoint(setup):
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


def test_openapi_endpoint_head_method(setup):
    """Check if REST API allows HEAD HTTP method for endpoint with OpenAPI specification."""
    response = client.head("/openapi.json")
    assert response.status_code == requests.codes.ok
    assert response.text == ""


def test_openapi_content(setup):
    """Check if the pre-generated OpenAPI schema is up-to date."""
    # retrieve pre-generated OpenAPI schema
    with open("docs/openapi.json", "r") as fin:
        pre_generated_schema = json.load(fin)

    # retrieve current OpenAPI schema
    response = client.get("/openapi.json")
    assert response.status_code == requests.codes.ok
    current_schema = response.json()

    # remove node that is not included in pre-generated OpenAPI schema
    del current_schema["info"]["license"]

    # compare schemas (as dicts)
    assert (
        current_schema == pre_generated_schema
    ), "Pre-generated schema is not up to date. Fix it with `make schema`."
