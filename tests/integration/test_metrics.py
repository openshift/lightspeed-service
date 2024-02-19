"""Integration tests for metrics exposed by the service."""

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


def retrieve_metrics(client):
    """Retrieve all service metrics."""
    response = client.get("/metrics/")

    # check that the /metrics/ endpoint is correct and we got
    # some response
    assert response.status_code == requests.codes.ok
    assert response.text is not None

    # return response text (it is not JSON!)
    return response.text


def test_metrics():
    """Check if service provides metrics endpoint with some expected counters."""
    response_text = retrieve_metrics(client)

    # counters that are expected to be part of metrics
    expected_counters = (
        "rest_api_calls_total",
        "llm_calls_total",
        "llm_calls_failures_total",
        "llm_validation_errors_total",
    )

    # check if all counters are present
    for expected_counter in expected_counters:
        assert f"{expected_counter} " in response_text


def get_counter_value(client, counter_name, path, status_code):
    """Retrieve counter value from metrics."""
    # counters with labels have the following format:
    # rest_api_calls_total{status_code="200"} 1.0
    # rest_api_calls_total{path="/openapi.json",status_code="200"} 1.0
    prefix = f'{counter_name}{{path="{path}",status_code="{status_code}"}} '

    response_text = retrieve_metrics(client)
    lines = [line.strip() for line in response_text.split("\n")]

    # try to find the given counter
    for line in lines:
        if line.startswith(prefix):
            line = line[len(prefix) :]
            # parse as float, convert that float to integer
            return int(float(line))
    raise Exception(f"Counter {counter_name} was not found in metrics")


def test_rest_api_call_counter_ok_status():
    """Check if REST API call counter works as expected, label with 200 OK status."""
    endpoint = "/readiness"

    # initialize counter with label by calling endpoint
    client.get(endpoint)
    old = get_counter_value(client, "rest_api_calls_total", endpoint, "200")

    # call some REST API endpoint
    client.get(endpoint)
    new = get_counter_value(client, "rest_api_calls_total", endpoint, "200")

    # compare counters
    assert new == old + 1, "Counter has not been updated properly"


def test_rest_api_call_counter_not_found_status():
    """Check if REST API call counter works as expected, label with 404 NotFound status."""
    endpoint = "/this-does-not-exists"

    # initialize counter with label
    client.get(endpoint)
    old = get_counter_value(client, "rest_api_calls_total", endpoint, "404")

    # call some REST API endpoint
    client.get(endpoint)
    new = get_counter_value(client, "rest_api_calls_total", endpoint, "404")

    # compare counters
    # just the NotFound value should change
    assert new == old + 1, "Counter for 404 NotFound  has not been updated properly"
