"""Integration tests for basic OLS REST API endpoints."""

import json
import os
import sys

import pytest
import requests

import ols.utils.suid as suid
import tests.e2e.cluster_utils as cluster_utils
import tests.e2e.helper_utils as testutils
import tests.e2e.metrics_utils as metrics_utils
from ols.constants import INVALID_QUERY_RESP, NO_RAG_CONTENT_RESP
from scripts.validate_response import ResponseValidation
from tests.e2e.constants import (
    BASIC_ENDPOINTS_TIMEOUT,
    CONVERSATION_ID,
    EVAL_THRESHOLD,
    LLM_REST_API_TIMEOUT,
    NON_LLM_REST_API_TIMEOUT,
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
    # TODO move cluster artifacts gathering(currently in utils.sh:must_gather()) to here.
    pass


@pytest.fixture(scope="module")
def response_eval(request):
    """Set response evaluation fixture."""
    with open("tests/test_data/question_answer_pair.json") as qna_f:
        qa_pairs = json.load(qna_f)

    eval_model = request.config.option.eval_model
    print(f"eval model: {eval_model}")

    return qa_pairs[eval_model]


def get_eval_question_answer(qna_pair, qna_id, scenario="without_rag"):
    """Get Evaluation question answer."""
    eval_query = qna_pair[scenario][qna_id]["question"]
    eval_answer = qna_pair[scenario][qna_id]["answer"]
    print(f"Evaluation question: {eval_query}")
    print(f"Ground truth answer: {eval_answer}")
    return eval_query, eval_answer


def test_readiness():
    """Test handler for /readiness REST API endpoint."""
    endpoint = "/readiness"
    with metrics_utils.RestAPICallCounterChecker(metrics_client, endpoint):
        response = client.get(endpoint, timeout=BASIC_ENDPOINTS_TIMEOUT)
        assert response.status_code == requests.codes.ok
        assert response.json() == {"status": {"status": "healthy"}}


def test_liveness():
    """Test handler for /liveness REST API endpoint."""
    endpoint = "/liveness"
    with metrics_utils.RestAPICallCounterChecker(metrics_client, endpoint):
        response = client.get(endpoint, timeout=BASIC_ENDPOINTS_TIMEOUT)
        assert response.status_code == requests.codes.ok
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
        print(vars(r))
        response = r.json()

        assert r.status_code == requests.codes.ok
        assert response["conversation_id"] == cid
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
        print(vars(response))
        assert response.status_code == requests.codes.ok

        expected_json = {
            "conversation_id": cid,
            "response": INVALID_QUERY_RESP,
            "referenced_documents": [],
            "truncated": False,
        }
        assert response.json() == expected_json


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
        print(vars(response))
        assert response.status_code == requests.codes.unprocessable_entity
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
        print(vars(response))
        assert response.status_code == requests.codes.unprocessable_entity
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
        json_response = response.json()

        # new conversation ID should be returned
        assert (
            "conversation_id" in json_response
        ), "New conversation ID was not generated"
        assert suid.check_suid(json_response["conversation_id"]), (
            "Conversation ID is not in UUID format" ""
        )


@pytest.mark.rag
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
        print(vars(response))
        assert response.status_code == requests.codes.ok
        json_response = response.json()

        # checking a few major information from response
        assert json_response["conversation_id"] == cid
        assert (
            "Kubernetes is" in json_response["response"]
            or "Kubernetes: It is" in json_response["response"]
        )
        assert (
            "orchestration tool" in json_response["response"]
            or "orchestration system" in json_response["response"]
            or "orchestration platform" in json_response["response"]
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


@pytest.mark.rag
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
        print(vars(response))
        assert response.status_code == requests.codes.ok
        json_response = response.json()
        assert "conversation_id" in json_response
        assert len(json_response["referenced_documents"]) > 0
        assert "performance" in json_response["referenced_documents"][0]
        assert "https://" in json_response["referenced_documents"][0]

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
        print(vars(response))
        assert response.status_code == requests.codes.ok
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
        print(vars(response))
        assert response.status_code == requests.codes.ok
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


@pytest.mark.cluster
def test_improper_token():
    """Test accessing /v1/query endpoint using improper auth. token."""
    response = client.post(
        "/v1/query",
        json={"query": "what is foo in bar?"},
        timeout=NON_LLM_REST_API_TIMEOUT,
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert response.status_code == requests.codes.forbidden


@pytest.mark.cluster
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


def test_feedback() -> None:
    """Check if feedback is properly stored.

    This is a full end-to-end scenario where the feedback is stored,
    retrieved and removed at the end (to avoid leftovers).
    """
    # check if feedback is enabled
    response = client.get("/v1/feedback/status", timeout=BASIC_ENDPOINTS_TIMEOUT)
    assert response.status_code == requests.codes.ok
    assert response.json()["status"]["enabled"] is True

    # check the feedback store is empty
    empty_feedback = client.get("/v1/feedback/list", timeout=BASIC_ENDPOINTS_TIMEOUT)
    assert empty_feedback.status_code == requests.codes.ok
    assert "feedbacks" in empty_feedback.json()
    assert len(empty_feedback.json()["feedbacks"]) == 0

    # store the feedback
    posted_feedback = client.post(
        "/v1/feedback",
        json={
            "conversation_id": CONVERSATION_ID,
            "user_question": "what is OCP4?",
            "llm_response": "Openshift 4 is ...",
            "sentiment": 1,
        },
        timeout=BASIC_ENDPOINTS_TIMEOUT,
    )
    assert posted_feedback.status_code == requests.codes.ok
    assert posted_feedback.json() == {"response": "feedback received"}

    # check the feedback store has one feedback
    stored_feedback = client.get("/v1/feedback/list", timeout=BASIC_ENDPOINTS_TIMEOUT)
    assert stored_feedback.status_code == requests.codes.ok
    assert "feedbacks" in stored_feedback.json()
    assert len(stored_feedback.json()["feedbacks"]) == 1

    # remove the feedback
    remove_feedback = client.delete(
        f'/v1/feedback/{stored_feedback.json()["feedbacks"][0]}',
        timeout=BASIC_ENDPOINTS_TIMEOUT,
    )
    assert remove_feedback.status_code == requests.codes.ok
    assert remove_feedback.json() == {"response": "feedback removed"}

    # check the feedback store is empty again
    removed_feedback = client.get("/v1/feedback/list", timeout=BASIC_ENDPOINTS_TIMEOUT)
    assert removed_feedback.status_code == requests.codes.ok
    assert "feedbacks" in removed_feedback.json()
    assert len(removed_feedback.json()["feedbacks"]) == 0


@pytest.mark.cluster
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
