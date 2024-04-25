"""Integration tests for basic OLS REST API endpoints."""

import json
import os
import pickle
import re
import shutil
import sys
import time
from pathlib import Path

import pytest
import requests

import ols.utils.suid as suid
import tests.e2e.cluster_utils as cluster_utils
import tests.e2e.helper_utils as testutils
import tests.e2e.metrics_utils as metrics_utils
from ols.constants import (
    HTTP_REQUEST_HEADERS_TO_REDACT,
    INVALID_QUERY_RESP,
    NO_RAG_CONTENT_RESP,
)
from scripts.validate_response import ResponseValidation
from tests.e2e.constants import (
    BASIC_ENDPOINTS_TIMEOUT,
    CONVERSATION_ID,
    EVAL_THRESHOLD,
    LLM_REST_API_TIMEOUT,
    NON_LLM_REST_API_TIMEOUT,
)

from .postgres_utils import (
    read_conversation_history,
    read_conversation_history_count,
    retrieve_connection,
)

# on_cluster is set to true when the tests are being run
# against ols running on a cluster
on_cluster = False

# OLS_URL env only needs to be set when running against a local ols instance,
# when ols is run against a cluster the url is retrieved from the cluster.
ols_url = os.getenv("OLS_URL", "http://localhost:8080")
if "localhost" not in ols_url:
    on_cluster = True

# generic http client for talking to OLS, when OLS is run on a cluster
# this client will be preconfigured with a valid user token header.
client = None
metrics_client = None


# constant from tests/config/cluster_install/ols_manifests.yaml
OLS_USER_DATA_PATH = "/app-root/ols-user-data"
OLS_USER_DATA_COLLECTION_INTERVAL = 10
OLS_COLLECTOR_DISABLING_FILE = OLS_USER_DATA_PATH + "/disable_collector"


def setup_module(module):
    """Set up common artifacts used by all e2e tests."""
    try:
        global ols_url, client, metrics_client
        token = None
        metrics_token = None
        if on_cluster:
            print("Setting up for on cluster test execution\n")
            ols_url = cluster_utils.get_ols_url("ols")
            cluster_utils.create_user("test-user")
            cluster_utils.create_user("metrics-test-user")
            token = cluster_utils.get_user_token("test-user")
            metrics_token = cluster_utils.get_user_token("metrics-test-user")
            cluster_utils.grant_sa_user_access("test-user", "ols-user")
            cluster_utils.grant_sa_user_access("metrics-test-user", "ols-metrics-user")
        else:
            print("Setting up for standalone test execution\n")

        client = testutils.get_http_client(ols_url, token)
        metrics_client = testutils.get_http_client(ols_url, metrics_token)
    except Exception as e:
        print(f"Failed to setup ols access: {e}")
        sys.exit(1)


def teardown_module(module):
    """Clean up the environment after all tests are executed."""
    # TODO: OLS-506 Move the program logic to gather cluster artifacts from utils.sh into e2e module
    pass


@pytest.fixture(scope="module")
def postgres_connection():
    """Fixture with Postgres connection."""
    return retrieve_connection()


@pytest.fixture(scope="module")
def response_eval(request):
    """Set response evaluation fixture."""
    with open("tests/test_data/question_answer_pair.json") as qna_f:
        qa_pairs = json.load(qna_f)

    eval_model = "gpt" if "gpt" in request.config.option.eval_model else "granite"
    print(f"eval model: {eval_model}")

    return qa_pairs[eval_model]


def get_eval_question_answer(qna_pair, qna_id, scenario="without_rag"):
    """Get Evaluation question answer."""
    eval_query = qna_pair[scenario][qna_id]["question"]
    eval_answer = qna_pair[scenario][qna_id]["answer"]
    print(f"Evaluation question: {eval_query}")
    print(f"Ground truth answer: {eval_answer}")
    return eval_query, eval_answer


def check_content_type(response, content_type):
    """Check if response content-type is set to defined value."""
    assert response.headers["content-type"].startswith(content_type)


def test_readiness():
    """Test handler for /readiness REST API endpoint."""
    endpoint = "/readiness"
    with metrics_utils.RestAPICallCounterChecker(metrics_client, endpoint):
        response = client.get(endpoint, timeout=BASIC_ENDPOINTS_TIMEOUT)
        assert response.status_code == requests.codes.ok
        check_content_type(response, "application/json")
        assert response.json() == {"status": {"status": "healthy"}}


def test_liveness():
    """Test handler for /liveness REST API endpoint."""
    endpoint = "/liveness"
    with metrics_utils.RestAPICallCounterChecker(metrics_client, endpoint):
        response = client.get(endpoint, timeout=BASIC_ENDPOINTS_TIMEOUT)
        assert response.status_code == requests.codes.ok
        check_content_type(response, "application/json")
        assert response.json() == {"status": {"status": "healthy"}}


def test_raw_prompt():
    """Check the REST API /v1/debug/query with POST HTTP method when expected payload is posted."""
    endpoint = "/v1/debug/query"
    with metrics_utils.RestAPICallCounterChecker(metrics_client, endpoint):
        cid = suid.get_suid()
        r = client.post(
            endpoint,
            json={
                "conversation_id": cid,
                "query": "respond to this message with the word hello",
            },
            timeout=LLM_REST_API_TIMEOUT,
        )
        assert r.status_code == requests.codes.ok

        check_content_type(r, "application/json")
        print(vars(r))
        response = r.json()

        assert response["conversation_id"] == cid
        assert response["referenced_documents"] == []
        assert "hello" in response["response"].lower()


def test_invalid_question():
    """Check the REST API /v1/query with POST HTTP method for invalid question."""
    endpoint = "/v1/query"
    with metrics_utils.RestAPICallCounterChecker(metrics_client, endpoint):
        cid = suid.get_suid()
        response = client.post(
            endpoint,
            json={"conversation_id": cid, "query": "test query"},
            timeout=LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.ok

        check_content_type(response, "application/json")
        print(vars(response))

        expected_json = {
            "conversation_id": cid,
            "response": INVALID_QUERY_RESP,
            "referenced_documents": [],
            "truncated": False,
        }
        assert response.json() == expected_json


def test_invalid_question_without_conversation_id():
    """Check the REST API /v1/query with invalid question and without conversation ID."""
    endpoint = "/v1/query"
    with metrics_utils.RestAPICallCounterChecker(metrics_client, endpoint):
        response = client.post(
            endpoint,
            json={"query": "test query"},
            timeout=LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.ok

        check_content_type(response, "application/json")
        print(vars(response))

        json_response = response.json()
        assert json_response["response"] == INVALID_QUERY_RESP
        assert json_response["referenced_documents"] == []
        assert json_response["truncated"] is False

        # new conversation ID should be generated
        assert suid.check_suid(json_response["conversation_id"]), (
            "Conversation ID is not in UUID format" ""
        )


def test_query_call_without_payload():
    """Check the REST API /v1/query with POST HTTP method when no payload is provided."""
    endpoint = "/v1/query"
    with metrics_utils.RestAPICallCounterChecker(
        metrics_client, endpoint, status_code=requests.codes.unprocessable_entity
    ):
        response = client.post(
            endpoint,
            timeout=LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.unprocessable_entity

        check_content_type(response, "application/json")
        print(vars(response))
        # the actual response might differ when new Pydantic version will be used
        # so let's do just primitive check
        assert "missing" in response.text


def test_query_call_with_improper_payload():
    """Check the REST API /v1/query with POST HTTP method when improper payload is provided."""
    endpoint = "/v1/query"
    with metrics_utils.RestAPICallCounterChecker(
        metrics_client, endpoint, status_code=requests.codes.unprocessable_entity
    ):
        response = client.post(
            endpoint,
            json={"parameter": "this-is-not-proper-question-my-friend"},
            timeout=NON_LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.unprocessable_entity

        check_content_type(response, "application/json")
        print(vars(response))
        # the actual response might differ when new Pydantic version will be used
        # so let's do just primitive check
        assert "missing" in response.text


def test_valid_question_improper_conversation_id(response_eval) -> None:
    """Check the REST API /v1/query with POST HTTP method for improper conversation ID."""
    endpoint = "/v1/query"
    eval_query, _ = get_eval_question_answer(response_eval, "eval1", "with_rag")

    with metrics_utils.RestAPICallCounterChecker(
        metrics_client, endpoint, status_code=requests.codes.internal_server_error
    ):
        response = client.post(
            endpoint,
            json={"conversation_id": "not-uuid", "query": eval_query},
            timeout=LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.internal_server_error

        check_content_type(response, "application/json")
        json_response = response.json()
        expected_response = {
            "detail": {
                "response": "Error retrieving conversation history",
                "cause": "Invalid conversation ID not-uuid",
            }
        }
        assert json_response == expected_response


def test_valid_question_missing_conversation_id(response_eval) -> None:
    """Check the REST API /v1/query with POST HTTP method for missing conversation ID."""
    endpoint = "/v1/query"
    eval_query, _ = get_eval_question_answer(response_eval, "eval1")

    with metrics_utils.RestAPICallCounterChecker(
        metrics_client, endpoint, status_code=requests.codes.ok
    ):
        response = client.post(
            endpoint,
            json={"conversation_id": "", "query": eval_query},
            timeout=LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.ok

        check_content_type(response, "application/json")
        json_response = response.json()

        # new conversation ID should be returned
        assert (
            "conversation_id" in json_response
        ), "New conversation ID was not generated"
        assert suid.check_suid(json_response["conversation_id"]), (
            "Conversation ID is not in UUID format" ""
        )


def test_too_long_question(response_eval) -> None:
    """Check the REST API /v1/query with too long question."""
    endpoint = "/v1/query"
    eval_query, _ = get_eval_question_answer(response_eval, "eval1", "without_rag")
    # let's make the query really large, larger that context window size
    eval_query *= 10000

    with metrics_utils.RestAPICallCounterChecker(
        metrics_client, endpoint, status_code=requests.codes.request_entity_too_large
    ):
        cid = suid.get_suid()
        response = client.post(
            endpoint,
            json={"conversation_id": cid, "query": eval_query},
            timeout=LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.request_entity_too_large

        check_content_type(response, "application/json")
        print(vars(response))
        json_response = response.json()
        assert "detail" in json_response
        assert json_response["detail"]["response"] == "Prompt is too long"


@pytest.mark.rag()
def test_valid_question(response_eval) -> None:
    """Check the REST API /v1/query with POST HTTP method for valid question and no yaml."""
    endpoint = "/v1/query"
    eval_query, eval_answer = get_eval_question_answer(
        response_eval, "eval1", "with_rag"
    )

    with metrics_utils.RestAPICallCounterChecker(metrics_client, endpoint):
        cid = suid.get_suid()
        response = client.post(
            endpoint,
            json={"conversation_id": cid, "query": eval_query},
            timeout=LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.ok

        check_content_type(response, "application/json")
        print(vars(response))
        json_response = response.json()

        # checking a few major information from response
        assert json_response["conversation_id"] == cid
        assert "Kubernetes is" in json_response["response"]
        assert re.search(
            r"orchestration (tool|system|platform|engine)",
            json_response["response"],
            re.IGNORECASE,
        )
        assert NO_RAG_CONTENT_RESP not in json_response["response"]

        score = ResponseValidation().get_similarity_score(
            json_response["response"], eval_answer
        )
        assert score <= EVAL_THRESHOLD


def test_valid_question_tokens_counter() -> None:
    """Check how the tokens counter are updated accordingly."""
    model, provider = metrics_utils.get_enabled_model_and_provider(metrics_client)

    endpoint = "/v1/query"
    with (
        metrics_utils.RestAPICallCounterChecker(metrics_client, endpoint),
        metrics_utils.TokenCounterChecker(metrics_client, model, provider),
    ):
        response = client.post(
            endpoint,
            json={"query": "what is kubernetes?"},
            timeout=LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.ok
        check_content_type(response, "application/json")


def test_invalid_question_tokens_counter() -> None:
    """Check how the tokens counter are updated accordingly."""
    model, provider = metrics_utils.get_enabled_model_and_provider(metrics_client)

    endpoint = "/v1/query"
    with (
        metrics_utils.RestAPICallCounterChecker(metrics_client, endpoint),
        metrics_utils.TokenCounterChecker(metrics_client, model, provider),
    ):
        response = client.post(
            endpoint,
            json={"query": "test query"},
            timeout=LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.ok
        check_content_type(response, "application/json")


def test_token_counters_for_query_call_without_payload() -> None:
    """Check how the tokens counter are updated accordingly."""
    model, provider = metrics_utils.get_enabled_model_and_provider(metrics_client)

    endpoint = "/v1/query"
    with (
        metrics_utils.RestAPICallCounterChecker(
            metrics_client, endpoint, status_code=requests.codes.unprocessable_entity
        ),
        metrics_utils.TokenCounterChecker(
            metrics_client,
            model,
            provider,
            expect_sent_change=False,
            expect_received_change=False,
        ),
    ):
        response = client.post(
            endpoint,
            timeout=LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.unprocessable_entity
        check_content_type(response, "application/json")


def test_token_counters_for_query_call_with_improper_payload() -> None:
    """Check how the tokens counter are updated accordingly."""
    model, provider = metrics_utils.get_enabled_model_and_provider(metrics_client)

    endpoint = "/v1/query"
    with (
        metrics_utils.RestAPICallCounterChecker(
            metrics_client, endpoint, status_code=requests.codes.unprocessable_entity
        ),
        metrics_utils.TokenCounterChecker(
            metrics_client,
            model,
            provider,
            expect_sent_change=False,
            expect_received_change=False,
        ),
    ):
        response = client.post(
            endpoint,
            json={"parameter": "this-is-not-proper-question-my-friend"},
            timeout=LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.unprocessable_entity
        check_content_type(response, "application/json")


@pytest.mark.rag()
def test_rag_question(response_eval) -> None:
    """Ensure responses include rag references."""
    endpoint = "/v1/query"
    eval_query, eval_answer = get_eval_question_answer(
        response_eval, "eval2", "with_rag"
    )

    with metrics_utils.RestAPICallCounterChecker(metrics_client, endpoint):
        response = client.post(
            endpoint,
            json={"query": eval_query},
            timeout=LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.ok
        check_content_type(response, "application/json")

        print(vars(response))
        json_response = response.json()
        assert "conversation_id" in json_response
        assert len(json_response["referenced_documents"]) > 0
        assert "virt" in json_response["referenced_documents"][0]["docs_url"]
        assert "https://" in json_response["referenced_documents"][0]["docs_url"]
        assert json_response["referenced_documents"][0]["title"]

        assert NO_RAG_CONTENT_RESP not in json_response["response"]

        score = ResponseValidation().get_similarity_score(
            json_response["response"], eval_answer
        )
        assert score <= EVAL_THRESHOLD


def test_query_filter() -> None:
    """Ensure responses does not include filtered words."""
    endpoint = "/v1/query"
    with metrics_utils.RestAPICallCounterChecker(metrics_client, endpoint):
        response = client.post(
            endpoint,
            json={"query": "what is foo in bar?"},
            timeout=LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.ok
        check_content_type(response, "application/json")

        print(vars(response))
        json_response = response.json()
        assert "conversation_id" in json_response

        # values to be filtered and replaced are defined in:
        # tests/config/singleprovider.e2e.template.config.yaml
        assert "openshift" in json_response["response"].lower()
        assert "deploy" in json_response["response"].lower()
        assert "foo" not in json_response["response"]
        assert "bar" not in json_response["response"]


def test_conversation_history() -> None:
    """Ensure conversations include previous query history."""
    endpoint = "/v1/query"
    with metrics_utils.RestAPICallCounterChecker(metrics_client, endpoint):
        response = client.post(
            endpoint,
            json={
                "query": "what is ingress in kubernetes?",
            },
            timeout=LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.ok
        check_content_type(response, "application/json")

        print(vars(response))
        json_response = response.json()
        assert "ingress" in json_response["response"].lower()

        # get the conversation id so we can reuse it for the follow up question
        cid = json_response["conversation_id"]
        response = client.post(
            endpoint,
            json={"conversation_id": cid, "query": "what?"},
            timeout=LLM_REST_API_TIMEOUT,
        )
        print(vars(response))
        assert response.status_code == requests.codes.ok
        json_response = response.json()
        assert "ingress" in json_response["response"].lower()


def test_query_with_provider_but_not_model(response_eval) -> None:
    """Check the REST API /v1/query with POST HTTP method for provider specified, but no model."""
    endpoint = "/v1/query"
    eval_query, _ = get_eval_question_answer(response_eval, "eval1")

    with metrics_utils.RestAPICallCounterChecker(
        metrics_client, endpoint, status_code=requests.codes.unprocessable_entity
    ):
        # just the provider is explicitly specified, but model selection is missing
        response = client.post(
            endpoint,
            json={"conversation_id": "", "query": eval_query, "provider": "bam"},
            timeout=LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.unprocessable_entity
        check_content_type(response, "application/json")

        json_response = response.json()

        # error thrown on Pydantic level
        assert (
            json_response["detail"][0]["msg"]
            == "Value error, LLM model must be specified when the provider is specified."
        )


def test_query_with_model_but_not_provider(response_eval) -> None:
    """Check the REST API /v1/query with POST HTTP method for model specified, but no provider."""
    endpoint = "/v1/query"
    eval_query, _ = get_eval_question_answer(response_eval, "eval1")

    with metrics_utils.RestAPICallCounterChecker(
        metrics_client, endpoint, status_code=requests.codes.unprocessable_entity
    ):
        # just model is explicitly specified, but provider selection is missing
        response = client.post(
            endpoint,
            json={
                "conversation_id": "",
                "query": eval_query,
                "model": "ibm/granite-13b-chat-v2",
            },
            timeout=LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.unprocessable_entity
        check_content_type(response, "application/json")

        json_response = response.json()

        assert (
            json_response["detail"][0]["msg"]
            == "Value error, LLM provider must be specified when the model is specified."
        )


def test_query_with_unknown_provider(response_eval) -> None:
    """Check the REST API /v1/query with POST HTTP method for unknown provider specified."""
    endpoint = "/v1/query"
    eval_query, _ = get_eval_question_answer(response_eval, "eval1")

    # retrieve currently selected model
    model, _ = metrics_utils.get_enabled_model_and_provider(metrics_client)

    with metrics_utils.RestAPICallCounterChecker(
        metrics_client, endpoint, status_code=requests.codes.unprocessable_entity
    ):
        # provider is unknown
        response = client.post(
            endpoint,
            json={
                "conversation_id": "",
                "query": eval_query,
                "provider": "foo",
                "model": model,
            },
            timeout=LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.unprocessable_entity
        check_content_type(response, "application/json")

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


def test_query_with_unknown_model(response_eval) -> None:
    """Check the REST API /v1/query with POST HTTP method for unknown model specified."""
    endpoint = "/v1/query"
    eval_query, _ = get_eval_question_answer(response_eval, "eval1")

    # retrieve currently selected provider
    _, provider = metrics_utils.get_enabled_model_and_provider(metrics_client)

    with metrics_utils.RestAPICallCounterChecker(
        metrics_client, endpoint, status_code=requests.codes.unprocessable_entity
    ):
        # model is unknown
        response = client.post(
            endpoint,
            json={
                "conversation_id": "",
                "query": eval_query,
                "provider": provider,
                "model": "bar",
            },
            timeout=LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.unprocessable_entity
        check_content_type(response, "application/json")

        json_response = response.json()

        # explicit response and cause check
        assert (
            "detail" in json_response
        ), "Improper response format: 'detail' node is missing"
        assert "Unable to process this request" in json_response["detail"]["response"]
        assert "Model 'bar' is not a valid model " in json_response["detail"]["cause"]


def test_metrics() -> None:
    """Check if service provides metrics endpoint with expected metrics."""
    response = metrics_client.get("/metrics", timeout=BASIC_ENDPOINTS_TIMEOUT)
    assert response.status_code == requests.codes.ok
    assert response.text is not None

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

    # check if all counters are present
    for expected_counter in expected_counters:
        assert f"{expected_counter} " in response.text

    # check the duration histogram presence
    assert 'response_duration_seconds_count{path="/metrics"}' in response.text
    assert 'response_duration_seconds_sum{path="/metrics"}' in response.text


def test_model_provider():
    """Read configured model and provider from metrics."""
    model, provider = metrics_utils.get_enabled_model_and_provider(metrics_client)

    # enabled model must be one of our expected combinations
    assert model, provider in {
        ("gpt-3.5-turbo", "openai"),
        ("gpt-3.5-turbo", "azure_openai"),
        ("ibm/granite-13b-chat-v2", "bam"),
        ("ibm/granite-13b-chat-v2", "watsonx"),
    }


def test_one_default_model_provider():
    """Check if one model and provider is selected as default."""
    states = metrics_utils.get_enable_status_for_all_models(metrics_client)
    enabled_states = [state for state in states if state is True]
    assert (
        len(enabled_states) == 1
    ), "one model and provider should be selected as default"


@pytest.mark.cluster()
def test_improper_token():
    """Test accessing /v1/query endpoint using improper auth. token."""
    response = client.post(
        "/v1/query",
        json={"query": "what is foo in bar?"},
        timeout=NON_LLM_REST_API_TIMEOUT,
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert response.status_code == requests.codes.forbidden


@pytest.mark.cluster()
def test_forbidden_user():
    """Test scenarios where we expect an unauthorized response.

    Test accessing /v1/query endpoint using the metrics user w/ no ols permissions,
    Test accessing /metrics endpoint using the ols user w/ no ols-metrics permissions.
    """
    response = metrics_client.post(
        "/v1/query",
        json={"query": "what is foo in bar?"},
        timeout=NON_LLM_REST_API_TIMEOUT,
    )
    assert response.status_code == requests.codes.forbidden
    response = client.get("/metrics", timeout=BASIC_ENDPOINTS_TIMEOUT)
    assert response.status_code == requests.codes.forbidden


@pytest.mark.cluster()
def test_feedback_can_post_with_wrong_token():
    """Test posting feedback with improper auth. token."""
    response = client.post(
        "/v1/feedback",
        json={
            "conversation_id": CONVERSATION_ID,
            "user_question": "what is OCP4?",
            "llm_response": "Openshift 4 is ...",
            "sentiment": 1,
        },
        timeout=BASIC_ENDPOINTS_TIMEOUT,
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert response.status_code == requests.codes.forbidden


@pytest.mark.standalone()
def test_feedback_storing_standalone():
    """Test if the feedbacks are stored properly."""
    # the standalone testing exposes the value via env
    feedback_dir = Path(os.environ["FEEDBACK_STORAGE_LOCATION"])

    # as this test is ran multiple times in test suite, we need to
    # ensure the storage is empty
    if feedback_dir.exists():
        shutil.rmtree(feedback_dir)

    response = client.post(
        "/v1/feedback",
        json={
            "conversation_id": CONVERSATION_ID,
            "user_question": "what is OCP4?",
            "llm_response": "Openshift 4 is ...",
            "sentiment": 1,
        },
        timeout=BASIC_ENDPOINTS_TIMEOUT,
    )

    assert response.status_code == requests.codes.ok

    assert feedback_dir.exists()

    feedbacks = list(feedback_dir.glob("*.json"))
    assert len(feedbacks) == 1

    feedback = feedbacks[0]
    with open(feedback) as f:
        feedback_data = json.load(f)

    assert feedback_data["user_id"]  # we don't care about actual value
    assert feedback_data["conversation_id"] == CONVERSATION_ID
    assert feedback_data["user_question"] == "what is OCP4?"
    assert feedback_data["llm_response"] == "Openshift 4 is ..."
    assert feedback_data["sentiment"] == 1


@pytest.mark.cluster()
def test_feedback_storing_cluster():
    """Test if the feedbacks are stored properly."""
    try:
        feedbacks_path = OLS_USER_DATA_PATH + "/feedback"
        pod_name = cluster_utils.get_single_existing_pod_name()

        # disable collector script to avoid interference with the test
        cluster_utils.create_file(pod_name, OLS_COLLECTOR_DISABLING_FILE, "")

        # there are multiple tests running agains cluster, so transcripts
        # can be already present - we need to ensure the storage is empty
        # for this test
        feedbacks = cluster_utils.list_path(pod_name, feedbacks_path)
        if feedbacks:
            cluster_utils.remove_dir(pod_name, feedbacks_path)
            assert cluster_utils.list_path(pod_name, feedbacks_path) == []

        response = client.post(
            "/v1/feedback",
            json={
                "conversation_id": CONVERSATION_ID,
                "user_question": "what is OCP4?",
                "llm_response": "Openshift 4 is ...",
                "sentiment": 1,
            },
            timeout=BASIC_ENDPOINTS_TIMEOUT,
        )

        assert response.status_code == requests.codes.ok

        feedback_data = cluster_utils.get_single_existing_feedback(
            pod_name, feedbacks_path
        )

        assert feedback_data["user_id"]  # we don't care about actual value
        assert feedback_data["conversation_id"] == CONVERSATION_ID
        assert feedback_data["user_question"] == "what is OCP4?"
        assert feedback_data["llm_response"] == "Openshift 4 is ..."
        assert feedback_data["sentiment"] == 1

    finally:
        # ensure script is enabled again after test (succesfull or not)
        cluster_utils.remove_file(pod_name, OLS_COLLECTOR_DISABLING_FILE)
        assert "disable_collector" not in cluster_utils.list_path(
            pod_name, OLS_USER_DATA_PATH
        )


def check_missing_field_response(response, field_name):
    """Check if 'Field required' error is returned by the service."""
    # error should be detected on Pydantic level
    assert response.status_code == requests.codes.unprocessable

    # the resonse payload should be valid JSON
    check_content_type(response, "application/json")
    json_response = response.json()

    # check payload details
    assert (
        "detail" in json_response
    ), "Improper response format: 'detail' node is missing"
    assert json_response["detail"][0]["msg"] == "Field required"
    assert json_response["detail"][0]["loc"][0] == "body"
    assert json_response["detail"][0]["loc"][1] == field_name


def test_feedback_missing_conversation_id():
    """Test posting feedback with missing conversation ID."""
    response = client.post(
        "/v1/feedback",
        json={
            "user_question": "what is OCP4?",
            "llm_response": "Openshift 4 is ...",
            "sentiment": 1,
        },
        timeout=BASIC_ENDPOINTS_TIMEOUT,
    )

    check_missing_field_response(response, "conversation_id")


def test_feedback_missing_user_question():
    """Test posting feedback with missing user question."""
    response = client.post(
        "/v1/feedback",
        json={
            "conversation_id": CONVERSATION_ID,
            "llm_response": "Openshift 4 is ...",
            "sentiment": 1,
        },
        timeout=BASIC_ENDPOINTS_TIMEOUT,
    )

    check_missing_field_response(response, "user_question")


def test_feedback_missing_llm_response():
    """Test posting feedback with missing LLM response."""
    response = client.post(
        "/v1/feedback",
        json={
            "conversation_id": CONVERSATION_ID,
            "user_question": "what is OCP4?",
            "sentiment": 1,
        },
        timeout=BASIC_ENDPOINTS_TIMEOUT,
    )

    check_missing_field_response(response, "llm_response")


def test_feedback_improper_conversation_id():
    """Test posting feedback with improper conversation ID."""
    response = client.post(
        "/v1/feedback",
        json={
            "conversation_id": "incorrect-conversation-id",
            "user_question": "what is OCP4?",
            "llm_response": "Openshift 4 is ...",
            "sentiment": 1,
        },
        timeout=BASIC_ENDPOINTS_TIMEOUT,
    )

    # error should be detected on Pydantic level
    assert response.status_code == requests.codes.unprocessable

    # for incorrect conversation ID, the payload should be valid JSON
    check_content_type(response, "application/json")
    json_response = response.json()

    assert (
        "detail" in json_response
    ), "Improper response format: 'detail' node is missing"
    assert (
        json_response["detail"][0]["msg"]
        == "Value error, Improper conversation ID incorrect-conversation-id"
    )


@pytest.mark.standalone()
def test_transcripts_storing_standalone():
    """Test if the transcripts are stored properly."""
    # the standalone testing exposes the value via env
    transcript_dir = Path(os.environ["TRANSCRIPTS_STORAGE_LOCATION"])

    # as this test is ran multiple times in test suite, we need to
    # ensure the storage is empty
    if transcript_dir.exists():
        shutil.rmtree(transcript_dir)

    query = "what is kubernetes?"

    response = client.post(
        "/v1/query",
        json={"query": query},
        timeout=LLM_REST_API_TIMEOUT,
    )

    assert response.status_code == requests.codes.ok

    assert transcript_dir.exists()

    transcripts = list(transcript_dir.glob("*/*/*.json"))
    assert len(transcripts) == 1

    transcript = transcripts[0]
    with open(transcript) as f:
        transcript_data = json.load(f)

    # we just test metadata exists as we don't know uuid for user and conversation
    assert transcript_data["metadata"]
    assert transcript_data["redacted_query"] == query
    assert transcript_data["query_is_valid"] is True
    assert transcript_data["llm_response"]  # we don't care about the content
    assert transcript_data["referenced_documents"] == []
    assert transcript_data["truncated"] is False


@pytest.mark.cluster()
def test_transcripts_storing_cluster():
    """Test if the transcripts are stored properly."""
    try:
        transcripts_path = OLS_USER_DATA_PATH + "/transcripts"
        pod_name = cluster_utils.get_single_existing_pod_name()

        # disable collector script to avoid interference with the test
        cluster_utils.create_file(pod_name, OLS_COLLECTOR_DISABLING_FILE, "")

        # there are multiple tests running agains cluster, so transcripts
        # can be already present - we need to ensure the storage is empty
        # for this test
        transcripts = cluster_utils.list_path(pod_name, transcripts_path)
        if transcripts:
            cluster_utils.remove_dir(pod_name, transcripts_path)
            assert cluster_utils.list_path(pod_name, transcripts_path) == []

        response = client.post(
            "/v1/query",
            json={
                "query": "what is kubernetes?",
            },
            timeout=LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.ok

        transcript = cluster_utils.get_single_existing_transcript(
            pod_name, transcripts_path
        )

        assert transcript["metadata"]  # just check if it is not empty
        assert transcript["redacted_query"] == "what is kubernetes?"
        # we don't want llm response influence this test
        assert "query_is_valid" in transcript
        assert "llm_response" in transcript
        assert "referenced_documents" in transcript
        assert transcript["referenced_documents"][0]["docs_url"]
        assert transcript["referenced_documents"][0]["title"]
        assert "truncated" in transcript
    finally:
        # ensure script is enabled again after test (succesfull or not)
        cluster_utils.remove_file(pod_name, OLS_COLLECTOR_DISABLING_FILE)
        assert "disable_collector" not in cluster_utils.list_path(
            pod_name, OLS_USER_DATA_PATH
        )


def test_openapi_endpoint():
    """Test handler for /opanapi REST API endpoint."""
    response = client.get("/openapi.json", timeout=BASIC_ENDPOINTS_TIMEOUT)
    assert response.status_code == requests.codes.ok
    check_content_type(response, "application/json")

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


def test_cache_existence(postgres_connection):
    """Test the cache existence."""
    if postgres_connection is None:
        pytest.skip("Postgres is not accessible." "")
        return

    value = read_conversation_history_count(postgres_connection)
    # check if history exists at all
    assert value is not None


def _perform_query(client, conversation_id, response_eval, qna_pair):
    endpoint = "/v1/query"
    eval_query, eval_answer = get_eval_question_answer(
        response_eval, qna_pair, "without_rag"
    )

    response = client.post(
        endpoint,
        json={"conversation_id": conversation_id, "query": eval_query},
        timeout=LLM_REST_API_TIMEOUT,
    )
    check_content_type(response, "application/json")
    print(vars(response))


def test_conversation_in_postgres_cache(response_eval, postgres_connection) -> None:
    """Check how/if the conversation is stored in cache."""
    if postgres_connection is None:
        pytest.skip("Postgres is not accessible." "")
        return

    cid = suid.get_suid()
    _perform_query(client, cid, response_eval, "eval1")

    conversation, updated_at = read_conversation_history(postgres_connection, cid)
    assert conversation is not None

    # unpickle conversation into list of messages
    unpickled = pickle.loads(conversation, errors="strict")  # noqa S301
    assert unpickled is not None

    # we expect one question + one answer
    assert len(unpickled) == 2

    # question check
    assert "what is kubernetes?" in unpickled[0].content

    # trivial check for answer (exact check is done in different tests)
    assert "Kubernetes" in unpickled[1].content

    # second question
    _perform_query(client, cid, response_eval, "eval2")

    conversation, updated_at = read_conversation_history(postgres_connection, cid)
    assert conversation is not None

    # unpickle conversation into list of messages
    unpickled = pickle.loads(conversation, errors="strict")  # noqa S301
    assert unpickled is not None

    # we expect one question + one answer
    assert len(unpickled) == 4

    # first question
    assert "what is kubernetes?" in unpickled[0].content

    # first answer
    assert "Kubernetes" in unpickled[1].content

    # second question
    assert "what is openshift virtualization?" in unpickled[2].content

    # second answer
    assert "OpenShift" in unpickled[3].content


@pytest.mark.cluster()
def test_user_data_collection():
    """Test user data collection.

    It is performed by checking the user data collection container logs
    for the expected messages in logs.
    A bit of trick is required to check just the logs since the last
    action (as container logs can be influenced by other tests).
    """

    def filter_logs(logs: str, last_log_line: str) -> str:
        filtered_logs = []
        new_logs = False
        for line in logs.split("\n"):
            if line == last_log_line:
                new_logs = True
                continue
            if new_logs:
                filtered_logs.append(line)
        return "\n".join(filtered_logs)

    def get_last_log_line(logs: str) -> str:
        return [line for line in logs.split("\n") if line][-1]

    # constants from tests/config/cluster_install/ols_manifests.yaml
    data_collection_container_name = "ols-sidecar-user-data-collector"
    pod_name = cluster_utils.get_single_existing_pod_name()

    # there are multiple tests running agains cluster, so user data
    # can be already present - we need to ensure the storage is empty
    # for this test
    user_data = cluster_utils.list_path(pod_name, OLS_USER_DATA_PATH)
    if user_data:
        cluster_utils.remove_dir(pod_name, OLS_USER_DATA_PATH + "/feedback")
        cluster_utils.remove_dir(pod_name, OLS_USER_DATA_PATH + "/transcripts")
        assert cluster_utils.list_path(pod_name, OLS_USER_DATA_PATH) == []

    # safety wait for the script to start after being disabled by other
    # tests
    time.sleep(OLS_USER_DATA_COLLECTION_INTERVAL + 1)

    # data shoud be pruned now and this is the point from which we want
    # to check the logs
    container_log = cluster_utils.get_container_log(
        pod_name, data_collection_container_name
    )
    last_log_line = get_last_log_line(container_log)

    # wait the collection period for some extra to give the script a
    # chance to log what we want to check
    time.sleep(OLS_USER_DATA_COLLECTION_INTERVAL + 1)

    # we just check that there are no data and the script is working
    container_log = cluster_utils.get_container_log(
        pod_name, data_collection_container_name
    )
    logs = filter_logs(container_log, last_log_line)

    assert "collected" not in logs
    assert "data uploaded with request_id:" not in logs
    assert "uploaded data removed" not in logs
    assert "data upload failed with response:" not in logs
    assert "contains no data, nothing to do..." in logs

    # get the log point for the next check
    last_log_line = get_last_log_line(container_log)

    # create a new data via feedback endpoint
    response = client.post(
        "/v1/feedback",
        json={
            "conversation_id": CONVERSATION_ID,
            "user_question": "what is OCP4?",
            "llm_response": "Openshift 4 is ...",
            "sentiment": 1,
        },
        timeout=BASIC_ENDPOINTS_TIMEOUT,
    )
    assert response.status_code == requests.codes.ok
    time.sleep(OLS_USER_DATA_COLLECTION_INTERVAL + 1)

    # check that data was packaged, sent and removed
    container_log = cluster_utils.get_container_log(
        pod_name, data_collection_container_name
    )
    logs = filter_logs(container_log, last_log_line)
    assert "collected 1 files from" in logs
    assert "data uploaded with request_id:" in logs
    assert "uploaded data removed" in logs
    assert "data upload failed with response:" not in logs
    user_data = cluster_utils.list_path(pod_name, OLS_USER_DATA_PATH + "/feedback/")
    assert user_data == []


@pytest.mark.cluster()
def test_http_header_redaction():
    """Test that sensitive HTTP headers are redacted from the logs."""
    for header in HTTP_REQUEST_HEADERS_TO_REDACT:
        endpoint = "/liveness"
        with metrics_utils.RestAPICallCounterChecker(metrics_client, endpoint):
            response = client.get(
                endpoint,
                headers={f"{header}": "some_value"},
                timeout=BASIC_ENDPOINTS_TIMEOUT,
            )
            assert response.status_code == requests.codes.ok
            check_content_type(response, "application/json")
            assert response.json() == {"status": {"status": "healthy"}}

    container_log = cluster_utils.get_container_log(
        cluster_utils.get_single_existing_pod_name(), "ols"
    )

    for header in HTTP_REQUEST_HEADERS_TO_REDACT:
        assert f'"{header}":"XXXXX"' in container_log
        assert f'"{header}":"some_value"' not in container_log
