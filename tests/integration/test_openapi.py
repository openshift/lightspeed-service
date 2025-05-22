"""Integration tests for REST API endpoint that provides OpenAPI specification."""

# we add new attributes into pytest instance, which is not recognized
# properly by linters
# pyright: reportAttributeAccessIssue=false

import json
import os
from unittest.mock import patch

import pytest
import requests
from fastapi.testclient import TestClient

from ols import config
from ols.constants import CONFIGURATION_FILE_NAME_ENV_VARIABLE
from ols.customize import metadata


@pytest.fixture(scope="function", autouse=True)
def _setup():
    """Setups the test client."""
    config.reload_from_yaml_file("tests/config/config_for_integration_tests.yaml")

    # we need to patch the config file path to point to the test
    # config file before we import anything from main.py
    with patch.dict(
        os.environ,
        {
            CONFIGURATION_FILE_NAME_ENV_VARIABLE: "tests/config/config_for_integration_tests.yaml"
        },
    ):
        # app.main need to be imported after the configuration is read
        from ols.app.main import app  # pylint: disable=C0415

        pytest.client = TestClient(app)


def test_openapi_endpoint():
    """Check if REST API provides endpoint with OpenAPI specification."""
    response = pytest.client.get("/openapi.json")
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
    assert f"{metadata.SERVICE_NAME} service API specification" in info["description"]

    # elementary check that all mandatory endpoints are covered
    paths = payload["paths"]
    for endpoint in ("/readiness", "/liveness", "/v1/query", "/v1/feedback"):
        assert endpoint in paths, f"Endpoint {endpoint} is not described"


def test_openapi_content():
    """Check if the pre-generated OpenAPI schema is up-to date."""
    # retrieve pre-generated OpenAPI schema
    with open("docs/openapi.json", encoding="utf-8") as fin:
        pre_generated_schema = json.load(fin)

    # retrieve current OpenAPI schema
    response = pytest.client.get("/openapi.json")
    assert response.status_code == requests.codes.ok
    current_schema = response.json()

    # remove node that is not included in pre-generated OpenAPI schema
    del current_schema["info"]["license"]

    # compare schemas (as dicts)
    assert (
        current_schema == pre_generated_schema
    ), "Pre-generated schema is not up to date. Fix it with `make schema`."
