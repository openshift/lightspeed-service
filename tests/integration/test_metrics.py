"""Integration tests for metrics exposed by the service."""

# we add new attributes into pytest instance, which is not recognized
# properly by linters
# pyright: reportAttributeAccessIssue=false

import logging
import os
import re
from unittest.mock import patch

import pytest
import requests
from fastapi.testclient import TestClient

from ols import config
from ols.app.models.config import LoggingConfig
from ols.constants import CONFIGURATION_FILE_NAME_ENV_VARIABLE
from ols.utils.logging_configurator import configure_logging

# counters that are expected to be part of metrics
expected_counters = (
    "ols_rest_api_calls_total",
    "ols_llm_calls_total",
    "ols_llm_calls_failures_total",
    "ols_llm_validation_errors_total",
    "ols_llm_token_sent_total",
    "ols_llm_token_received_total",
    "ols_provider_model_configuration",
)


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


def retrieve_metrics(client):
    """Retrieve all service metrics."""
    response = pytest.client.get("/metrics")

    # check that the /metrics endpoint is correct and we got
    # some response
    assert response.status_code == requests.codes.ok
    assert response.text is not None

    # return response text (it is not JSON!)
    return response.text


def test_metrics():
    """Check if service provides metrics endpoint with some expected counters."""
    response_text = retrieve_metrics(pytest.client)

    # check if all counters are present
    for expected_counter in expected_counters:
        assert (
            f"{expected_counter} " in response_text
        ), f"Counter {expected_counter} not found in {response_text}"


def test_metrics_with_debug_log(caplog):
    """Check if service provides metrics endpoint with some expected counters."""
    logging_config = LoggingConfig(app_log_level="debug")

    configure_logging(logging_config)
    logger = logging.getLogger("ols")
    logger.handlers = [caplog.handler]  # add caplog handler to logger

    response_text = retrieve_metrics(pytest.client)

    # check if all counters are present
    for expected_counter in expected_counters:
        assert (
            f"{expected_counter} " in response_text
        ), f"Counter {expected_counter} not found in {response_text}"

    # check if the metrics are also found in the log
    captured_out = caplog.text
    for expected_counter in expected_counters:
        assert expected_counter in captured_out


def test_metrics_with_debug_logging_suppressed(caplog):
    """Check if service provides metrics endpoint with some counters and log output suppressed."""
    logging_config = LoggingConfig(app_log_level="debug")
    assert config.ols_config.logging_config is not None
    config.ols_config.logging_config.suppress_metrics_in_log = True

    configure_logging(logging_config)

    logger = logging.getLogger("ols")
    logger.handlers = [caplog.handler]  # add caplog handler to logger

    response_text = retrieve_metrics(pytest.client)

    # check if all counters are present
    for expected_counter in expected_counters:
        assert (
            f"{expected_counter} " in response_text
        ), f"Counter {expected_counter} not found in {response_text}"

    # check if the metrics are NOT found in the log
    captured_out = caplog.text
    for expected_counter in expected_counters:
        assert expected_counter not in captured_out


def get_counter_value(client, counter_name, path, status_code):
    """Retrieve counter value from metrics."""
    # counters with labels have the following format:
    # rest_api_calls_total{path="/openapi.json",status_code="200"} 1.0
    prefix = f'{counter_name}{{path="{path}",status_code="{status_code}"}} '

    response_text = retrieve_metrics(client)
    lines = [line.strip() for line in response_text.split("\n")]

    # try to find the given counter
    for line in lines:
        if line.startswith(prefix):
            without_prefix = line[len(prefix) :]
            # parse as float, convert that float to integer
            return int(float(without_prefix))
    raise Exception(f"Counter {counter_name} was not found in metrics")


def test_rest_api_call_counter_ok_status():
    """Check if REST API call counter works as expected, label with 200 OK status."""
    endpoint = "/liveness"

    # initialize counter with label by calling endpoint
    pytest.client.get(endpoint)
    old = get_counter_value(pytest.client, "ols_rest_api_calls_total", endpoint, "200")

    # call some REST API endpoint
    pytest.client.get(endpoint)
    new = get_counter_value(pytest.client, "ols_rest_api_calls_total", endpoint, "200")

    # compare counters
    assert new == old + 1, "Counter has not been updated properly"


def test_rest_api_call_counter_not_found_status():
    """Check if REST API call counter ignore metrics for non-existing endpoint."""
    endpoint = "/this-does-not-exists"

    # call the non-existent REST API endpoint
    pytest.client.get(endpoint)
    # expect Exception about not finding "ols_rest_api_calls_total" in metrics
    with pytest.raises(
        Exception, match="Counter ols_rest_api_calls_total was not found in metrics"
    ):
        get_counter_value(pytest.client, "ols_rest_api_calls_total", endpoint, "404")


def test_metrics_duration():
    """Check if service provides metrics for durations."""
    response_text = retrieve_metrics(pytest.client)

    # duration histograms are expected to be part of metrics

    # first: check summary and statistic
    assert 'response_duration_seconds_count{path="/metrics"}' in response_text
    assert 'response_duration_seconds_sum{path="/metrics"}' in response_text

    # second: check the histogram itself
    pattern = re.compile(
        r"response_duration_seconds_bucket{le=\"0\.[0-9]+\",path=\"\/metrics\"}"
    )
    # re.findall() returns empty list if not found, and this empty list is treated as False
    assert re.findall(pattern, response_text)
    pattern = re.compile(
        r"response_duration_seconds_bucket{le=\"\+Inf\",path=\"\/metrics\"}"
    )
    assert re.findall(pattern, response_text)


def test_provider_model_configuration_metrics():
    """Check if provider_model_configuration metrics shows the expected information."""
    response_text = retrieve_metrics(pytest.client)
    print(response_text)
    for provider in ("bam", "openai"):
        for model in ("m1", "m2"):
            if provider == "bam" and model == "m1":
                # default/enabled model should have a metric value of 1.0
                assert (
                    f'provider_model_configuration{{model="{model}",provider="{provider}"}} 1.0'
                    in response_text
                )
            else:
                # non-enabled models should have a metric value of 0.0
                assert (
                    f'provider_model_configuration{{model="{model}",provider="{provider}"}} 0.0'
                    in response_text
                )
