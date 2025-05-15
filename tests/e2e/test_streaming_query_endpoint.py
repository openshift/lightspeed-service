"""End to end tests for the REST API streming query endpoint."""

# we add new attributes into pytest instance, which is not recognized
# properly by linters
# pyright: reportAttributeAccessIssue=false

import json
import re

import pytest
import requests

from ols import constants
from ols.utils import suid
from tests.e2e.utils import cluster as cluster_utils
from tests.e2e.utils import metrics as metrics_utils
from tests.e2e.utils import response as response_utils
from tests.e2e.utils.decorators import retry

from . import test_api

STREAMING_QUERY_ENDPOINT = "/v1/streaming_query"


# NOTE: This approach forces the connection to close after the request,
# aligning with HTTP/1.0 behavior and potentially preventing incomplete
# chunked reads, that results in tests "flakiness".
def post_with_defaults(endpoint, **kwargs):
    """Send POST request with HTTP/1.0 header and timeout (if not in kwargs)."""
    return pytest.client.post(
        endpoint,
        headers={"Connection": "close"},
        timeout=kwargs.pop("timeout", test_api.LLM_REST_API_TIMEOUT),
        **kwargs,
    )


def parse_streaming_response_to_events(response: str) -> list[dict]:
    """Parse streaming response to events."""
    json_objects = [
        line.replace("data: ", "") for line in response.split("\n") if line.strip()
    ]
    json_array = "[" + ",".join(json_objects) + "]"
    return json.loads(json_array)


def construct_response_from_streamed_events(events: dict) -> str:
    """Construct response from streamed events."""
    response = ""
    for event in events:
        if event["event"] == "token":
            response += event["data"]["token"]
    return response


def test_invalid_question():
    """Check the endpoint POST method for invalid question."""
    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client, STREAMING_QUERY_ENDPOINT
    ):
        cid = suid.get_suid()

        response = post_with_defaults(
            STREAMING_QUERY_ENDPOINT,
            json={
                "conversation_id": cid,
                "query": "how to make burger?",
                "media_type": constants.MEDIA_TYPE_TEXT,
            },
        )

        assert response.status_code == requests.codes.ok
        response_utils.check_content_type(response, constants.MEDIA_TYPE_TEXT)

        assert re.search(
            r"(sorry|questions|assist)",
            response.text,
            re.IGNORECASE,
        )


def test_invalid_question_without_conversation_id():
    """Check the endpoint POST method for generating new conversation_id."""
    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client, STREAMING_QUERY_ENDPOINT
    ):
        response = post_with_defaults(
            STREAMING_QUERY_ENDPOINT,
            json={
                "query": "how to make burger?",
                "media_type": constants.MEDIA_TYPE_JSON,
            },
        )
        assert response.status_code == requests.codes.ok
        response_utils.check_content_type(response, constants.MEDIA_TYPE_JSON)
        events = parse_streaming_response_to_events(response.text)

        # new conversation ID should be generated
        assert events[0]["event"] == "start"
        assert events[0]["data"]
        assert suid.check_suid(events[0]["data"]["conversation_id"])


def test_query_call_without_payload():
    """Check the endpoint with POST HTTP method when no payload is provided."""
    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client,
        STREAMING_QUERY_ENDPOINT,
        status_code=requests.codes.unprocessable_entity,
    ):
        response = post_with_defaults(
            STREAMING_QUERY_ENDPOINT,
        )
        assert response.status_code == requests.codes.unprocessable_entity

        response_utils.check_content_type(response, constants.MEDIA_TYPE_JSON)
        # the actual response might differ when new Pydantic version
        # will be used so let's do just primitive check
        assert "missing" in response.text


def test_query_call_with_improper_payload():
    """Check the endpoint with POST HTTP method when improper payload is provided."""
    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client,
        STREAMING_QUERY_ENDPOINT,
        status_code=requests.codes.unprocessable_entity,
    ):
        response = post_with_defaults(
            STREAMING_QUERY_ENDPOINT,
            json={"parameter": "this-is-unknown-parameter"},
            timeout=test_api.NON_LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.unprocessable_entity

        response_utils.check_content_type(response, constants.MEDIA_TYPE_JSON)
        # the actual response might differ when new Pydantic version will be used
        # so let's do just primitive check
        assert "missing" in response.text


def test_valid_question_improper_conversation_id() -> None:
    """Check the endpoint with POST HTTP method for improper conversation ID."""
    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client,
        STREAMING_QUERY_ENDPOINT,
        status_code=requests.codes.internal_server_error,
    ):
        response = post_with_defaults(
            STREAMING_QUERY_ENDPOINT,
            json={"conversation_id": "not-uuid", "query": "what is kubernetes?"},
        )
        assert response.status_code == requests.codes.internal_server_error

        response_utils.check_content_type(response, constants.MEDIA_TYPE_JSON)
        json_response = response.json()
        expected_response = {
            "detail": {
                "response": "Error retrieving conversation history",
                "cause": "Invalid conversation ID not-uuid",
            }
        }
        assert json_response == expected_response


def test_too_long_question() -> None:
    """Check the endpoint with too long question."""
    # let's make the query really large, larger that context window size
    query = "what is kubernetes?" * 25000

    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client,
        STREAMING_QUERY_ENDPOINT,
        status_code=requests.codes.ok,
    ):
        cid = suid.get_suid()
        response = post_with_defaults(
            STREAMING_QUERY_ENDPOINT,
            json={
                "conversation_id": cid,
                "query": query,
                "media_type": constants.MEDIA_TYPE_JSON,
            },
        )
        assert response.status_code == requests.codes.ok

        response_utils.check_content_type(response, constants.MEDIA_TYPE_JSON)

        events = parse_streaming_response_to_events(response.text)

        assert len(events) == 2
        assert events[1]["event"] == "error"
        assert events[1]["data"]["response"] == "Prompt is too long"


@pytest.mark.smoketest
@pytest.mark.rag
def test_valid_question() -> None:
    """Check the endpoint with POST HTTP method for valid question and no yaml."""
    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client, STREAMING_QUERY_ENDPOINT
    ):
        cid = suid.get_suid()
        response = post_with_defaults(
            STREAMING_QUERY_ENDPOINT,
            json={"conversation_id": cid, "query": "what is kubernetes?"},
        )
        assert response.status_code == requests.codes.ok

        response_utils.check_content_type(response, constants.MEDIA_TYPE_TEXT)

        assert "Kubernetes is" in response.text
        assert re.search(
            r"orchestration (tool|system|platform|engine)",
            response.text,
            re.IGNORECASE,
        )


@pytest.mark.rag
def test_ocp_docs_version_same_as_cluster_version() -> None:
    """Check that the version of OCP docs matches the cluster we're on."""
    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client, STREAMING_QUERY_ENDPOINT
    ):
        cid = suid.get_suid()
        response = post_with_defaults(
            STREAMING_QUERY_ENDPOINT,
            json={
                "conversation_id": cid,
                "query": "welcome openshift container platform documentation",
                "media_type": constants.MEDIA_TYPE_JSON,
            },
        )
        assert response.status_code == requests.codes.ok

        response_utils.check_content_type(response, constants.MEDIA_TYPE_JSON)
        major, minor = cluster_utils.get_cluster_version()
        events = parse_streaming_response_to_events(response.text)
        assert events[-1]["event"] == "end"
        assert events[-1]["data"]["referenced_documents"]
        assert (
            f"{major}.{minor}"
            in events[-1]["data"]["referenced_documents"][0]["doc_url"]
        )


def test_valid_question_tokens_counter() -> None:
    """Check how the tokens counter are updated accordingly."""
    model, provider = metrics_utils.get_enabled_model_and_provider(
        pytest.metrics_client
    )

    with (
        metrics_utils.RestAPICallCounterChecker(
            pytest.metrics_client, STREAMING_QUERY_ENDPOINT
        ),
        metrics_utils.TokenCounterChecker(pytest.metrics_client, model, provider),
    ):
        response = post_with_defaults(
            STREAMING_QUERY_ENDPOINT,
            json={"query": "what is kubernetes?"},
        )
        assert response.status_code == requests.codes.ok
        response_utils.check_content_type(response, constants.MEDIA_TYPE_TEXT)


def test_invalid_question_tokens_counter() -> None:
    """Check how the tokens counter are updated accordingly."""
    model, provider = metrics_utils.get_enabled_model_and_provider(
        pytest.metrics_client
    )

    with (
        metrics_utils.RestAPICallCounterChecker(
            pytest.metrics_client, STREAMING_QUERY_ENDPOINT
        ),
        metrics_utils.TokenCounterChecker(pytest.metrics_client, model, provider),
    ):
        response = post_with_defaults(
            STREAMING_QUERY_ENDPOINT,
            json={"query": "how to make burger?"},
        )
        assert response.status_code == requests.codes.ok
        response_utils.check_content_type(response, constants.MEDIA_TYPE_TEXT)


def test_token_counters_for_query_call_without_payload() -> None:
    """Check how the tokens counter are updated accordingly."""
    model, provider = metrics_utils.get_enabled_model_and_provider(
        pytest.metrics_client
    )

    with (
        metrics_utils.RestAPICallCounterChecker(
            pytest.metrics_client,
            STREAMING_QUERY_ENDPOINT,
            status_code=requests.codes.unprocessable_entity,
        ),
        metrics_utils.TokenCounterChecker(
            pytest.metrics_client,
            model,
            provider,
            expect_sent_change=False,
            expect_received_change=False,
        ),
    ):
        response = post_with_defaults(
            STREAMING_QUERY_ENDPOINT,
        )
        assert response.status_code == requests.codes.unprocessable_entity
        response_utils.check_content_type(response, constants.MEDIA_TYPE_JSON)


def test_token_counters_for_query_call_with_improper_payload() -> None:
    """Check how the tokens counter are updated accordingly."""
    model, provider = metrics_utils.get_enabled_model_and_provider(
        pytest.metrics_client
    )

    with (
        metrics_utils.RestAPICallCounterChecker(
            pytest.metrics_client,
            STREAMING_QUERY_ENDPOINT,
            status_code=requests.codes.unprocessable_entity,
        ),
        metrics_utils.TokenCounterChecker(
            pytest.metrics_client,
            model,
            provider,
            expect_sent_change=False,
            expect_received_change=False,
        ),
    ):
        response = post_with_defaults(
            STREAMING_QUERY_ENDPOINT,
            json={"parameter": "this-is-not-proper-question-my-friend"},
        )
        assert response.status_code == requests.codes.unprocessable_entity
        response_utils.check_content_type(response, constants.MEDIA_TYPE_JSON)


@pytest.mark.tool_calling
@pytest.mark.rag
@retry(max_attempts=3, wait_between_runs=60)
def test_rag_question() -> None:
    """Ensure responses include rag references."""
    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client, STREAMING_QUERY_ENDPOINT
    ):
        response = post_with_defaults(
            STREAMING_QUERY_ENDPOINT,
            json={
                "query": "what is openshift virtualization?",
                "media_type": constants.MEDIA_TYPE_JSON,
            },
        )
        assert response.status_code == requests.codes.ok
        response_utils.check_content_type(response, constants.MEDIA_TYPE_JSON)

        events = parse_streaming_response_to_events(response.text)

        assert events[0]["event"] == "start"
        assert events[0]["data"]["conversation_id"]
        assert events[-1]["event"] == "end"
        ref_docs = events[-1]["data"]["referenced_documents"]
        assert ref_docs
        assert "virt" in ref_docs[0]["doc_url"]
        assert "https://" in ref_docs[0]["doc_url"]

        # ensure no duplicates in docs
        docs_urls = [doc["doc_url"] for doc in ref_docs]
        assert len(set(docs_urls)) == len(docs_urls)


@pytest.mark.cluster
def test_query_filter() -> None:
    """Ensure responses does not include filtered words and redacted words are not logged."""
    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client, STREAMING_QUERY_ENDPOINT
    ):
        query = "what is foo in bar?"
        response = post_with_defaults(
            STREAMING_QUERY_ENDPOINT,
            json={"query": query},
        )
        assert response.status_code == requests.codes.ok
        response_utils.check_content_type(response, constants.MEDIA_TYPE_TEXT)

        # values to be filtered and replaced are defined in:
        # tests/config/singleprovider.e2e.template.config.yaml
        response_text = response.text.lower()
        assert "openshift" in response_text
        assert "deploy" in response_text
        response_words = response_text.split()
        assert "foo" not in response_words
        assert "bar" not in response_words

        # Retrieve the pod name
        ols_container_name = "lightspeed-service-api"
        pod_name = cluster_utils.get_pod_by_prefix()[0]

        # Check if filtered words are redacted in the logs
        container_log = cluster_utils.get_container_log(pod_name, ols_container_name)

        # Ensure redacted patterns do not appear in the logs
        unwanted_patterns = ["foo ", "what is foo in bar?"]
        for line in container_log.splitlines():
            # Only check lines that are not part of a query
            if re.search(r'Body: \{"query":', line):
                continue
            # check that the pattern is indeed not found in logs
            for pattern in unwanted_patterns:
                assert pattern not in line.lower()

        # Ensure the intended redaction has occurred
        assert "what is deployment in openshift?" in container_log


@retry(max_attempts=3, wait_between_runs=10)
def test_conversation_history() -> None:
    """Ensure conversations include previous query history."""
    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client, STREAMING_QUERY_ENDPOINT
    ):
        response = post_with_defaults(
            STREAMING_QUERY_ENDPOINT,
            json={
                "query": "what is ingress in kubernetes?",
                "media_type": constants.MEDIA_TYPE_JSON,
            },
        )
        scenario_fail_msg = "First call to LLM without conversation history has failed"
        assert response.status_code == requests.codes.ok, scenario_fail_msg
        response_utils.check_content_type(
            response, constants.MEDIA_TYPE_JSON, scenario_fail_msg
        )

        events = parse_streaming_response_to_events(response.text)
        response_text = construct_response_from_streamed_events(events).lower()

        assert "ingress" in response_text, scenario_fail_msg

        # get the conversation id so we can reuse it for the follow up question
        assert events[0]["event"] == "start"
        cid = events[0]["data"]["conversation_id"]
        response = post_with_defaults(
            STREAMING_QUERY_ENDPOINT,
            json={
                "conversation_id": cid,
                "query": "what?",
                "media_type": constants.MEDIA_TYPE_JSON,
            },
        )

        scenario_fail_msg = "Second call to LLM with conversation history has failed"
        assert response.status_code == requests.codes.ok
        response_utils.check_content_type(
            response, constants.MEDIA_TYPE_JSON, scenario_fail_msg
        )

        events = parse_streaming_response_to_events(response.text)
        response_text = construct_response_from_streamed_events(events).lower()
        assert "ingress" in response_text, scenario_fail_msg


def test_query_with_provider_but_not_model() -> None:
    """Check the endpoint with POST HTTP method for provider specified, but no model."""
    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client,
        STREAMING_QUERY_ENDPOINT,
        status_code=requests.codes.unprocessable_entity,
    ):
        # just the provider is explicitly specified, but model selection is missing
        response = post_with_defaults(
            STREAMING_QUERY_ENDPOINT,
            json={
                "conversation_id": "",
                "query": "what is kubernetes?",
                "provider": "bam",
            },
        )
        assert response.status_code == requests.codes.unprocessable_entity
        response_utils.check_content_type(response, constants.MEDIA_TYPE_JSON)

        json_response = response.json()

        # error thrown on Pydantic level
        assert (
            json_response["detail"][0]["msg"]
            == "Value error, LLM model must be specified when the provider is specified."
        )


def test_query_with_model_but_not_provider() -> None:
    """Check the endpoint with POST HTTP method for model specified, but no provider."""
    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client,
        STREAMING_QUERY_ENDPOINT,
        status_code=requests.codes.unprocessable_entity,
    ):
        # just model is explicitly specified, but provider selection is missing
        response = post_with_defaults(
            STREAMING_QUERY_ENDPOINT,
            json={
                "conversation_id": "",
                "query": "what is kubernetes?",
                "model": "model-name",
            },
        )
        assert response.status_code == requests.codes.unprocessable_entity
        response_utils.check_content_type(response, constants.MEDIA_TYPE_JSON)

        json_response = response.json()

        assert (
            json_response["detail"][0]["msg"]
            == "Value error, LLM provider must be specified when the model is specified."
        )


def test_query_with_unknown_provider() -> None:
    """Check the endpoint with POST HTTP method for unknown provider specified."""
    # retrieve currently selected model
    model, _ = metrics_utils.get_enabled_model_and_provider(pytest.metrics_client)

    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client,
        STREAMING_QUERY_ENDPOINT,
        status_code=requests.codes.unprocessable_entity,
    ):
        # provider is unknown
        response = post_with_defaults(
            STREAMING_QUERY_ENDPOINT,
            json={
                "conversation_id": "",
                "query": "what is kubernetes?",
                "provider": "foo",
                "model": model,
            },
        )
        assert response.status_code == requests.codes.unprocessable_entity
        response_utils.check_content_type(response, constants.MEDIA_TYPE_JSON)

        json_response = response.json()

        # explicit response and cause check
        assert (
            "detail" in json_response
        ), "Improper response format: 'detail' node is missing"
        assert "Unable to process this request" in json_response["detail"]["response"]
        assert (
            "Provider 'foo' is not a valid provider."
            in json_response["detail"]["cause"]
        )


def test_query_with_unknown_model() -> None:
    """Check the endpoint with POST HTTP method for unknown model specified."""
    # retrieve currently selected provider
    _, provider = metrics_utils.get_enabled_model_and_provider(pytest.metrics_client)

    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client,
        STREAMING_QUERY_ENDPOINT,
        status_code=requests.codes.unprocessable_entity,
    ):
        # model is unknown
        response = post_with_defaults(
            STREAMING_QUERY_ENDPOINT,
            json={
                "conversation_id": "",
                "query": "what is kubernetes?",
                "provider": provider,
                "model": "bar",
            },
        )
        assert response.status_code == requests.codes.unprocessable_entity
        response_utils.check_content_type(response, constants.MEDIA_TYPE_JSON)

        json_response = response.json()

        # explicit response and cause check
        assert (
            "detail" in json_response
        ), "Improper response format: 'detail' node is missing"
        assert "Unable to process this request" in json_response["detail"]["response"]
        assert "Model 'bar' is not a valid model " in json_response["detail"]["cause"]


@pytest.mark.tool_calling
def test_tool_calling_text() -> None:
    """Check the endpoint for tool calling in text format."""
    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client, STREAMING_QUERY_ENDPOINT
    ):
        cid = suid.get_suid()
        response = post_with_defaults(
            STREAMING_QUERY_ENDPOINT,
            json={
                "conversation_id": cid,
                "query": "Get me current running pods in openshift-lightspeed namespace",
                "media_type": constants.MEDIA_TYPE_TEXT,
            },
        )
        assert response.status_code == requests.codes.ok

        response_utils.check_content_type(response, constants.MEDIA_TYPE_TEXT)

        # Sometime granite doesn't summarize well,
        # response may contain actual tool commands.
        assert re.search(
            r"(lightspeed-app-server|\[\"pods\", \"-n\", \"openshift-lightspeed\"\])",
            response.text.lower(),
        )


@pytest.mark.tool_calling
def test_tool_calling_events() -> None:
    """Check the endpoint for tool calling in event format."""
    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client, STREAMING_QUERY_ENDPOINT
    ):
        cid = suid.get_suid()
        response = post_with_defaults(
            STREAMING_QUERY_ENDPOINT,
            json={
                "conversation_id": cid,
                "query": "Get me current running pods in openshift-lightspeed namespace",
                "media_type": constants.MEDIA_TYPE_JSON,
            },
        )
        assert response.status_code == requests.codes.ok
        response_utils.check_content_type(response, constants.MEDIA_TYPE_JSON)

        events = parse_streaming_response_to_events(response.text)
        unique_events = {e["event"] for e in events}
        response_text = construct_response_from_streamed_events(events).lower()

        # Sometime granite doesn't summarize well,
        # response may contain actual tool commands.
        assert re.search(
            r"(lightspeed-app-server|\[\"pods\", \"-n\", \"openshift-lightspeed\"\])",
            response_text,
        )
        assert "tool_call" in unique_events
        assert "tool_result" in unique_events
