"""Utilities for reading and checking metrics from REST API."""

from typing import Optional

import requests

from tests.e2e.constants import BASIC_ENDPOINTS_TIMEOUT


def read_metrics(client):
    """Read all metrics using REST API call."""
    response = client.get("/metrics", timeout=BASIC_ENDPOINTS_TIMEOUT)

    # check that the /metrics endpoint is correct and we got
    # some response
    assert response.status_code == requests.codes.ok
    assert response.text is not None

    return response.text


def get_rest_api_counter_value(
    client, path, status_code=requests.codes.ok, default=None
):
    """Retrieve counter value from metrics."""
    response = read_metrics(client)
    counter_name = "rest_api_calls_total"

    # counters with labels have the following format:
    # rest_api_calls_total{path="/openapi.json",status_code="200"} 1.0
    prefix = f'{counter_name}{{path="{path}",status_code="{status_code}"}} '

    return get_counter_value(counter_name, prefix, response, default)


def get_response_duration_seconds_value(client, path, default=None):
    """Retrieve counter value from metrics."""
    response = read_metrics(client)
    counter_name = "response_duration_seconds_sum"

    # counters with response durations have the following format:
    # response_duration_seconds_sum{path="/v1/debug/query"} 0.123
    prefix = f'{counter_name}{{path="{path}"}} '

    return get_counter_value(counter_name, prefix, response, default, to_int=False)


def get_model_provider_counter_value(
    client, counter_name, model, provider, default=None
):
    """Retrieve counter value from metrics."""
    response = read_metrics(client)

    # counters with model and provider have the following format:
    # llm_token_sent_total{model="ibm/granite-13b-chat-v2",provider="bam"} 8.0
    # llm_token_received_total{model="ibm/granite-13b-chat-v2",provider="bam"} 2465.0
    prefix = f'{counter_name}{{model="{model}",provider="{provider}"}} '

    return get_counter_value(counter_name, prefix, response, default)


def get_metric_labels(lines, info_node_name) -> Optional[dict]:
    """Get labels associated with a metric string as printed from /metrics."""
    prefix = info_node_name

    attrs = {}
    for line in lines:
        if line.startswith(prefix):
            # strip prefix
            metric = line[len(prefix) + 1 :]

            # strip suffix
            labels = metric[: metric.find("} ")]
            labels = labels.split(",")
            for label in labels:
                kv = label.split("=")
                print(f"kv: {kv}")
                # strip leading/trailing quotation from value
                attrs[kv[0]] = kv[1][1:-1]
            return attrs

    # info node was not found
    return None


def get_model_and_provider(client):
    """Read configured model and provider from metrics."""
    response = read_metrics(client)
    lines = [line.strip() for line in response.split("\n")]

    labels = get_metric_labels(lines, "selected_model_info")

    return labels["model"], labels["provider"]


def get_counter_value(counter_name, prefix, response, default=None, to_int=True):
    """Try to retrieve counter value from response with all metrics."""
    lines = [line.strip() for line in response.split("\n")]

    # try to find the given counter
    for line in lines:
        if line.startswith(prefix):
            without_prefix = line[len(prefix) :]
            # parse counter value as float
            value = float(without_prefix)
            # convert that float to integer if needed
            if to_int:
                return int(value)
            return value

    # counter was not found, which might be ok for first API call
    if default is not None:
        return default

    raise Exception(f"Counter {counter_name} was not found in metrics")


def check_counter_increases(endpoint, old_counter, new_counter, delta=1):
    """Check if the counter value increases as expected."""
    assert (
        new_counter >= old_counter + delta
    ), f"REST API counter for {endpoint} has not been updated properly"


def check_duration_sum_increases(endpoint, old_counter, new_counter):
    """Check if the counter value with total duration increases as expected."""
    assert (
        new_counter > old_counter
    ), f"Duration sum for {endpoint} has not been updated properly"


def check_token_counter_increases(counter, old_counter, new_counter, expect_change):
    """Check if the counter value increases as expected."""
    if expect_change:
        assert (
            new_counter > old_counter
        ), f"Counter for {counter} tokens has not been updated properly"
    else:
        assert (
            new_counter == old_counter
        ), f"Counter for {counter} tokens has changed, which is unexpected"


class RestAPICallCounterChecker:
    """Context manager to check if REST API counter is increased for given endpoint."""

    def __init__(self, client, endpoint, status_code=requests.codes.ok):
        """Register client and endpoint."""
        self.client = client
        self.endpoint = endpoint
        self.status_code = status_code

    def __enter__(self):
        """Retrieve old counter value before calling REST API."""
        self.old_counter = get_rest_api_counter_value(
            self.client, self.endpoint, status_code=self.status_code, default=0
        )
        self.old_duration = get_response_duration_seconds_value(
            self.client, self.endpoint, default=0
        )

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Retrieve new counter value after calling REST API, and check if it increased."""
        # test if REST API endpoint counter has been updated
        new_counter = get_rest_api_counter_value(
            self.client, self.endpoint, status_code=self.status_code
        )
        check_counter_increases(self.endpoint, self.old_counter, new_counter)

        # test if duration counter has been updated
        new_duration = get_response_duration_seconds_value(
            self.client,
            self.endpoint,
        )
        check_duration_sum_increases(self.endpoint, self.old_duration, new_duration)


class TokenCounterChecker:
    """Context manager to check if token counters are increased before and after LLL calls.

    Example:
    ```python
    with TokenCounterChecker(client, "ibm/granite-13b-chat-v2", "bam"):
        ...
        ...
        ...
    """

    def __init__(
        self,
        client,
        model,
        provider,
        expect_sent_change=True,
        expect_received_change=True,
    ):
        """Register model and provider which tokens will be checked."""
        self.model = model
        self.provider = provider
        self.client = client
        # when model nor provider are specified (OLS cluster), don't run checks
        self.skip_check = model is None or provider is None

        # expect change in number of sent tokens
        self.expect_sent_change = expect_sent_change

        # expect change in number of received tokens
        self.expect_received_change = expect_received_change

    def __enter__(self):
        """Retrieve old counter values before calling LLM."""
        if self.skip_check:
            return
        self.old_counter_token_sent_total = get_model_provider_counter_value(
            self.client, "llm_token_sent_total", self.model, self.provider, default=0
        )
        self.old_counter_token_received_total = get_model_provider_counter_value(
            self.client,
            "llm_token_received_total",
            self.model,
            self.provider,
            default=0,
        )

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Retrieve new counter value after calling REST API, and check if it increased."""
        if self.skip_check:
            return
        # check if counter for sent tokens has been updated
        new_counter_token_sent_total = get_model_provider_counter_value(
            self.client, "llm_token_sent_total", self.model, self.provider
        )
        check_token_counter_increases(
            "sent",
            self.old_counter_token_sent_total,
            new_counter_token_sent_total,
            self.expect_sent_change,
        )

        # check if counter for received tokens has been updated
        new_counter_token_received_total = get_model_provider_counter_value(
            self.client,
            "llm_token_received_total",
            self.model,
            self.provider,
            default=0,
        )
        check_token_counter_increases(
            "received",
            self.old_counter_token_received_total,
            new_counter_token_received_total,
            self.expect_received_change,
        )
