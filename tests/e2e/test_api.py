"""Integration tests for basic OLS REST API endpoints."""

import os

import requests
from httpx import Client

url = os.getenv("OLS_URL", "http://localhost:8080")
token = os.getenv("OLS_TOKEN")
client = Client(base_url=url, verify=False)  # noqa: S501
if token:
    client.headers.update({"Authorization": f"Bearer {token}"})


conversation_id = "12345678-abcd-0000-0123-456789abcdef"


# timeout settings
BASIC_ENDPOINTS_TIMEOUT = 5
NON_LLM_REST_API_TIMEOUT = 20
LLM_REST_API_TIMEOUT = 90


def read_metrics(client):
    """Read all metrics using REST API call."""
    response = client.get("/metrics/", timeout=BASIC_ENDPOINTS_TIMEOUT)

    # check that the /metrics/ endpoint is correct and we got
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


def check_token_counter_increases(counter, old_counter, new_counter):
    """Check if the counter value increases as expected."""
    assert (
        new_counter > old_counter
    ), f"Counter for {counter} tokens has not been updated properly"


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

    def __init__(self, client, model, provider):
        """Register model and provider which tokens will be checked."""
        self.model = model
        self.provider = provider
        self.client = client

    def __enter__(self):
        """Retrieve old counter values before calling LLM."""
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
        # check if counter for sent tokens has been updated
        new_counter_token_sent_total = get_model_provider_counter_value(
            self.client, "llm_token_sent_total", self.model, self.provider
        )
        check_token_counter_increases(
            "sent", self.old_counter_token_sent_total, new_counter_token_sent_total
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
        )


def test_readiness():
    """Test handler for /readiness REST API endpoint."""
    endpoint = "/readiness"
    with RestAPICallCounterChecker(client, endpoint):
        response = client.get(endpoint, timeout=BASIC_ENDPOINTS_TIMEOUT)
        assert response.status_code == requests.codes.ok
        assert response.json() == {"status": {"status": "healthy"}}


def test_liveness():
    """Test handler for /liveness REST API endpoint."""
    endpoint = "/liveness"
    with RestAPICallCounterChecker(client, endpoint):
        response = client.get(endpoint, timeout=BASIC_ENDPOINTS_TIMEOUT)
        assert response.status_code == requests.codes.ok
        assert response.json() == {"status": {"status": "healthy"}}


def test_raw_prompt():
    """Check the REST API /v1/debug/query with POST HTTP method when expected payload is posted."""
    endpoint = "/v1/debug/query"
    with RestAPICallCounterChecker(client, endpoint):
        r = client.post(
            endpoint,
            json={
                "conversation_id": conversation_id,
                "query": "respond to this message with the word hello",
            },
            timeout=LLM_REST_API_TIMEOUT,
        )
        print(vars(r))
        response = r.json()

        assert r.status_code == requests.codes.ok
        assert response["conversation_id"] == conversation_id
        assert "hello" in response["response"].lower()


def test_invalid_question():
    """Check the REST API /v1/query with POST HTTP method for invalid question."""
    endpoint = "/v1/query"
    with RestAPICallCounterChecker(client, endpoint):
        response = client.post(
            endpoint,
            json={"conversation_id": conversation_id, "query": "test query"},
            timeout=LLM_REST_API_TIMEOUT,
        )
        print(vars(response))
        assert response.status_code == requests.codes.ok
        expected_details = (
            "I can only answer questions about OpenShift and Kubernetes. "
            "Please rephrase your question"
        )
        expected_json = {
            "conversation_id": conversation_id,
            "response": expected_details,
            "referenced_documents": [],
            "truncated": False,
        }
        assert response.json() == expected_json


def test_query_call_without_payload():
    """Check the REST API /v1/query with POST HTTP method when no payload is provided."""
    endpoint = "/v1/query"
    with RestAPICallCounterChecker(
        client, endpoint, status_code=requests.codes.unprocessable_entity
    ):
        response = client.post(
            endpoint,
            timeout=LLM_REST_API_TIMEOUT,
        )
        print(vars(response))
        assert response.status_code == requests.codes.unprocessable_entity
        # the actual response might differ when new Pydantic version will be used
        # so let's do just primitive check
        assert "missing" in response.text


def test_query_call_with_improper_payload():
    """Check the REST API /v1/query with POST HTTP method when improper payload is provided."""
    endpoint = "/v1/query"
    with RestAPICallCounterChecker(
        client, endpoint, status_code=requests.codes.unprocessable_entity
    ):
        response = client.post(
            endpoint,
            json={"parameter": "this-is-not-proper-question-my-friend"},
            timeout=NON_LLM_REST_API_TIMEOUT,
        )
        print(vars(response))
        assert response.status_code == requests.codes.unprocessable_entity
        # the actual response might differ when new Pydantic version will be used
        # so let's do just primitive check
        assert "missing" in response.text


def test_valid_question() -> None:
    """Check the REST API /v1/query with POST HTTP method for valid question and no yaml."""
    endpoint = "/v1/query"
    with RestAPICallCounterChecker(client, endpoint):
        response = client.post(
            endpoint,
            json={"conversation_id": conversation_id, "query": "what is kubernetes?"},
            timeout=LLM_REST_API_TIMEOUT,
        )
        print(vars(response))
        assert response.status_code == requests.codes.ok
        json_response = response.json()
        json_response["conversation_id"] == conversation_id
        # checking a few major information from response
        assert "Kubernetes is" in json_response["response"]
        assert (
            "orchestration tool" in json_response["response"]
            or "orchestration system" in json_response["response"]
            or "orchestration platform" in json_response["response"]
        )
        assert (
            "The following response was generated without access to reference content:"
            not in json_response["response"]
        )


def test_rag_question() -> None:
    """Ensure responses include rag references."""
    endpoint = "/v1/query"
    with RestAPICallCounterChecker(client, endpoint):
        response = client.post(
            endpoint,
            json={"query": "what is the first step to install an openshift cluster?"},
            timeout=LLM_REST_API_TIMEOUT,
        )
        print(vars(response))
        assert response.status_code == requests.codes.ok
        json_response = response.json()
        assert len(json_response["referenced_documents"]) > 0
        assert "install" in json_response["referenced_documents"][0]
        assert "https://" in json_response["referenced_documents"][0]

        assert (
            "The following response was generated without access to reference content:"
            not in json_response["response"]
        )


def test_query_filter() -> None:
    """Ensure responses does not include filtered words."""
    endpoint = "/v1/query"
    with RestAPICallCounterChecker(client, endpoint):
        response = client.post(
            endpoint,
            json={"query": "what is foo in bar?"},
            timeout=LLM_REST_API_TIMEOUT,
        )
        print(vars(response))
        assert response.status_code == requests.codes.ok
        json_response = response.json()
        assert len(json_response["referenced_documents"]) > 0
        assert "openshift" in json_response["referenced_documents"][0]
        assert "https://" in json_response["referenced_documents"][0]

        # values to be filtered and replaced are defined in:
        # tests/config/singleprovider.e2e.template.config.yaml
        assert "openshift" in json_response["response"].lower()
        assert "deployment" in json_response["response"].lower()
        assert "foo" not in json_response["response"]
        assert "bar" not in json_response["response"]


def test_conversation_history() -> None:
    """Ensure conversations include previous query history."""
    endpoint = "/v1/query"
    conversation_id = "12345678-abcd-0000-0123-356789abcdef"
    with RestAPICallCounterChecker(client, endpoint):
        response = client.post(
            endpoint,
            json={
                "conversation_id": conversation_id,
                "query": "what is ingress in kubernetes?",
            },
            timeout=LLM_REST_API_TIMEOUT,
        )
        print(vars(response))
        assert response.status_code == requests.codes.ok
        json_response = response.json()
        assert "ingress" in json_response["response"].lower()
        response = client.post(
            endpoint,
            json={"conversation_id": conversation_id, "query": "what?"},
            timeout=LLM_REST_API_TIMEOUT,
        )
        print(vars(response))
        assert response.status_code == requests.codes.ok
        json_response = response.json()
        assert "ingress" in json_response["response"].lower()


def test_metrics() -> None:
    """Check if service provides metrics endpoint with expected metrics."""
    response = client.get("/metrics/", timeout=BASIC_ENDPOINTS_TIMEOUT)
    assert response.status_code == requests.codes.ok
    assert response.text is not None

    # counters that are expected to be part of metrics
    expected_counters = (
        "rest_api_calls_total",
        "llm_calls_total",
        "llm_calls_failures_total",
        "llm_validation_errors_total",
        "llm_token_sent_total",
        "llm_token_received_total",
    )

    # check if all counters are present
    for expected_counter in expected_counters:
        assert f"{expected_counter} " in response.text

    # check the duration histogram presence
    assert 'response_duration_seconds_count{path="/metrics/"}' in response.text
    assert 'response_duration_seconds_sum{path="/metrics/"}' in response.text


def test_improper_token():
    """Test accessing /v1/query endpoint using improper auth. token."""
    # let's assume that auth. is enabled when token is specified
    if token:
        response = client.post(
            "/v1/query",
            json={"query": "what is foo in bar?"},
            timeout=NON_LLM_REST_API_TIMEOUT,
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == requests.codes.forbidden
