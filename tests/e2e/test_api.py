"""Integration tests for basic OLS REST API endpoints."""

import json
import os
import shutil
import sys
from pathlib import Path

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


# constant from tests/config/cluster_install/ols_manifests.yaml
OLS_USER_DATA_PATH = "/app-root/ols-user-data"


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


def check_content_type(response, content_type):
    """Check if response content-type is set to defined value."""
    assert response.headers["content-type"].startswith(content_type)


def test_readiness():
    """Test handler for /readiness REST API endpoint."""
    endpoint = "/readiness"
    with metrics_utils.RestAPICallCounterChecker(metrics_client, endpoint):
        response = client.get(endpoint, timeout=BASIC_ENDPOINTS_TIMEOUT)
        check_content_type(response, "application/json")
        assert response.status_code == requests.codes.ok
        assert response.json() == {"status": {"status": "healthy"}}


def test_liveness():
    """Test handler for /liveness REST API endpoint."""
    endpoint = "/liveness"
    with metrics_utils.RestAPICallCounterChecker(metrics_client, endpoint):
        response = client.get(endpoint, timeout=BASIC_ENDPOINTS_TIMEOUT)
        check_content_type(response, "application/json")
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
        check_content_type(r, "application/json")
        print(vars(r))
        response = r.json()

        assert r.status_code == requests.codes.ok
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
        check_content_type(response, "application/json")
        print(vars(response))
        assert response.status_code == requests.codes.ok

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
        check_content_type(response, "application/json")
        print(vars(response))
        assert response.status_code == requests.codes.ok

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
        check_content_type(response, "application/json")
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
        check_content_type(response, "application/json")
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
        check_content_type(response, "application/json")
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
        check_content_type(response, "application/json")
        assert response.status_code == requests.codes.ok
        json_response = response.json()

        # new conversation ID should be returned
        assert (
            "conversation_id" in json_response
        ), "New conversation ID was not generated"
        assert suid.check_suid(json_response["conversation_id"]), (
            "Conversation ID is not in UUID format" ""
        )


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
        check_content_type(response, "application/json")
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
        check_content_type(response, "application/json")
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
        check_content_type(response, "application/json")
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
        check_content_type(response, "application/json")
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
        check_content_type(response, "application/json")
        assert response.status_code == requests.codes.unprocessable_entity


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
        check_content_type(response, "application/json")
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
        check_content_type(response, "application/json")
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
        check_content_type(response, "application/json")
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
        check_content_type(response, "application/json")
        assert response.status_code == requests.codes.unprocessable_entity
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
        check_content_type(response, "application/json")
        assert response.status_code == requests.codes.unprocessable_entity
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
        check_content_type(response, "application/json")
        assert response.status_code == requests.codes.unprocessable_entity
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
        check_content_type(response, "application/json")
        assert response.status_code == requests.codes.unprocessable_entity
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
    # constant from tests/config/cluster_install/ols_manifests.yaml
    feedbacks_path = OLS_USER_DATA_PATH + "/feedback"
    pod_name = cluster_utils.get_single_existing_pod_name()

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

    feedback_data = cluster_utils.get_single_existing_feedback(pod_name, feedbacks_path)

    assert feedback_data["user_id"]  # we don't care about actual value
    assert feedback_data["conversation_id"] == CONVERSATION_ID
    assert feedback_data["user_question"] == "what is OCP4?"
    assert feedback_data["llm_response"] == "Openshift 4 is ..."
    assert feedback_data["sentiment"] == 1


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
    transcripts_path = OLS_USER_DATA_PATH + "/transcripts"
    pod_name = cluster_utils.get_single_existing_pod_name()

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
    assert "truncated" in transcript
