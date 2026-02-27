"""End to end tests for the REST API endpoint /query."""

# we add new attributes into pytest instance, which is not recognized
# properly by linters
# pyright: reportAttributeAccessIssue=false

import os
import re

import pytest
import requests

from ols.utils import suid
from tests.e2e.utils import cluster as cluster_utils
from tests.e2e.utils import metrics as metrics_utils
from tests.e2e.utils import response as response_utils
from tests.e2e.utils.decorators import retry

from . import test_api

QUERY_ENDPOINT = "/v1/query"


@pytest.mark.skip_with_lcore
def test_invalid_question():
    """Check the REST API /v1/query with POST HTTP method for invalid question."""
    with metrics_utils.RestAPICallCounterChecker(pytest.metrics_client, QUERY_ENDPOINT):
        cid = suid.get_suid()
        response = pytest.client.post(
            QUERY_ENDPOINT,
            json={"conversation_id": cid, "query": "how to make burger?"},
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.ok

        response_utils.check_content_type(response, "application/json")
        print(vars(response))

        json_response = response.json()
        assert json_response["conversation_id"] == cid
        assert json_response["referenced_documents"] == []
        assert json_response["input_tokens"] > 0
        assert json_response["output_tokens"] > 0
        assert not json_response["truncated"]
        # LLM shouldn't answer non-ocp queries or
        # at least acknowledges that query is non-ocp.
        # Below assert is minimal due to model randomness.
        assert re.search(
            r"(sorry|questions|assist)",
            json_response["response"],
            re.IGNORECASE,
        )


def test_invalid_question_without_conversation_id():
    """Check the REST API /v1/query with invalid question and without conversation ID."""
    with metrics_utils.RestAPICallCounterChecker(pytest.metrics_client, QUERY_ENDPOINT):
        response = pytest.client.post(
            QUERY_ENDPOINT,
            json={"query": "how to make burger?"},
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.ok

        response_utils.check_content_type(response, "application/json")
        print(vars(response))

        json_response = response.json()
        assert json_response["referenced_documents"] == []
        assert json_response["truncated"] is False
        assert json_response["input_tokens"] > 0
        assert json_response["output_tokens"] > 0
        # Query classification is disabled by default,
        # and we rely on the model (controlled by prompt) to reject non-ocp queries.
        # Randomness in response is expected.
        # assert json_response["response"] == INVALID_QUERY_RESP
        assert re.search(
            r"(sorry|questions|assist)",
            json_response["response"],
            re.IGNORECASE,
        )
        if os.getenv("LCORE", "False").lower() not in ("true", "1", "t"):
            # new conversation ID should be generated
            assert suid.check_suid(
                json_response["conversation_id"]
            ), "Conversation ID is not in UUID format"


def test_query_call_without_payload():
    """Check the REST API /v1/query with POST HTTP method when no payload is provided."""
    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client,
        QUERY_ENDPOINT,
        status_code=requests.codes.unprocessable_entity,
    ):
        response = pytest.client.post(
            QUERY_ENDPOINT,
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.unprocessable_entity

        response_utils.check_content_type(response, "application/json")
        print(vars(response))
        # the actual response might differ when new Pydantic version will be used
        # so let's do just primitive check
        assert "missing" in response.text


def test_query_call_with_improper_payload():
    """Check the REST API /v1/query with POST HTTP method when improper payload is provided."""
    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client,
        QUERY_ENDPOINT,
        status_code=requests.codes.unprocessable_entity,
    ):
        response = pytest.client.post(
            QUERY_ENDPOINT,
            json={"parameter": "this-is-not-proper-question-my-friend"},
            timeout=test_api.NON_LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.unprocessable_entity

        response_utils.check_content_type(response, "application/json")
        print(vars(response))
        # the actual response might differ when new Pydantic version will be used
        # so let's do just primitive check
        assert "missing" in response.text


@pytest.mark.skip_with_lcore
def test_valid_question_improper_conversation_id() -> None:
    """Check the REST API /v1/query with POST HTTP method for improper conversation ID."""
    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client,
        QUERY_ENDPOINT,
        status_code=requests.codes.internal_server_error,
    ):
        response = pytest.client.post(
            QUERY_ENDPOINT,
            json={"conversation_id": "not-uuid", "query": "what is kubernetes?"},
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.internal_server_error

        response_utils.check_content_type(response, "application/json")
        json_response = response.json()
        expected_response = {
            "detail": {
                "response": "Error retrieving conversation history",
                "cause": "Invalid conversation ID not-uuid",
            }
        }
        assert json_response == expected_response


@pytest.mark.skip_with_lcore
@retry(max_attempts=3, wait_between_runs=10)
def test_valid_question_missing_conversation_id() -> None:
    """Check the REST API /v1/query with POST HTTP method for missing conversation ID."""
    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client, QUERY_ENDPOINT, status_code=requests.codes.ok
    ):
        response = pytest.client.post(
            QUERY_ENDPOINT,
            json={"conversation_id": "", "query": "what is kubernetes?"},
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.ok

        response_utils.check_content_type(response, "application/json")
        json_response = response.json()

        # new conversation ID should be returned
        assert (
            "conversation_id" in json_response
        ), "New conversation ID was not generated"
        assert suid.check_suid(
            json_response["conversation_id"]
        ), "Conversation ID is not in UUID format"


@pytest.mark.skip_with_lcore
def test_too_long_question() -> None:
    """Check the REST API /v1/query with too long question."""
    # let's make the query really large, larger that context window size
    query = "what is kubernetes?" * 25000

    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client,
        QUERY_ENDPOINT,
        status_code=requests.codes.request_entity_too_large,
    ):
        cid = suid.get_suid()
        response = pytest.client.post(
            QUERY_ENDPOINT,
            json={"conversation_id": cid, "query": query},
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.request_entity_too_large

        response_utils.check_content_type(response, "application/json")
        print(vars(response))
        json_response = response.json()
        assert "detail" in json_response
        assert json_response["detail"]["response"] == "Prompt is too long"


@pytest.mark.smoketest
@pytest.mark.rag
def test_valid_question() -> None:
    """Check the REST API /v1/query with POST HTTP method for valid question and no yaml."""
    with metrics_utils.RestAPICallCounterChecker(pytest.metrics_client, QUERY_ENDPOINT):
        if os.getenv("LCORE", "False").lower() not in ("true", "1", "t"):
            cid = suid.get_suid()
            response = pytest.client.post(
                QUERY_ENDPOINT,
                json={
                    "conversation_id": cid,
                    "query": "what is kubernetes in the context of OpenShift?",
                },
                timeout=test_api.LLM_REST_API_TIMEOUT,
            )
            assert response.status_code == requests.codes.ok

            response_utils.check_content_type(response, "application/json")
            print(vars(response))
            json_response = response.json()

            # checking a few major information from response
            assert json_response["conversation_id"] == cid
        else:
            response = pytest.client.post(
                QUERY_ENDPOINT,
                json={
                    "query": "what is kubernetes in the context of OpenShift?",
                },
                timeout=test_api.LLM_REST_API_TIMEOUT,
            )
            assert response.status_code == requests.codes.ok

            response_utils.check_content_type(response, "application/json")
            print(vars(response))
            json_response = response.json()
        assert re.search(
            r"kubernetes|openshift",
            json_response["response"],
            re.IGNORECASE,
        )
        assert json_response["input_tokens"] > 0
        assert json_response["output_tokens"] > 0


@pytest.mark.rag
def test_ocp_docs_version_same_as_cluster_version() -> None:
    """Check that the version of OCP docs matches the cluster we're on."""
    with metrics_utils.RestAPICallCounterChecker(pytest.metrics_client, QUERY_ENDPOINT):
        cid = suid.get_suid()
        response = pytest.client.post(
            QUERY_ENDPOINT,
            json={
                "conversation_id": cid,
                "query": "welcome openshift container platform documentation",
            },
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.ok

        response_utils.check_content_type(response, "application/json")
        print(vars(response))
        json_response = response.json()

        major, minor = cluster_utils.get_cluster_version()

        assert len(json_response["referenced_documents"]) > 1
        assert f"{major}.{minor}" in json_response["referenced_documents"][0]["doc_url"]


@pytest.mark.skip_with_lcore
def test_valid_question_tokens_counter() -> None:
    """Check how the tokens counter are updated accordingly."""
    model, provider = metrics_utils.get_enabled_model_and_provider(
        pytest.metrics_client
    )

    with (
        metrics_utils.RestAPICallCounterChecker(pytest.metrics_client, QUERY_ENDPOINT),
        metrics_utils.TokenCounterChecker(pytest.metrics_client, model, provider),
    ):
        response = pytest.client.post(
            QUERY_ENDPOINT,
            json={"query": "what is kubernetes?"},
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.ok
        response_utils.check_content_type(response, "application/json")


@pytest.mark.skip_with_lcore
def test_invalid_question_tokens_counter() -> None:
    """Check how the tokens counter are updated accordingly."""
    model, provider = metrics_utils.get_enabled_model_and_provider(
        pytest.metrics_client
    )

    with (
        metrics_utils.RestAPICallCounterChecker(pytest.metrics_client, QUERY_ENDPOINT),
        metrics_utils.TokenCounterChecker(pytest.metrics_client, model, provider),
    ):
        response = pytest.client.post(
            QUERY_ENDPOINT,
            json={"query": "how to make burger?"},
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.ok
        response_utils.check_content_type(response, "application/json")


@pytest.mark.skip_with_lcore
def test_token_counters_for_query_call_without_payload() -> None:
    """Check how the tokens counter are updated accordingly."""
    model, provider = metrics_utils.get_enabled_model_and_provider(
        pytest.metrics_client
    )

    with (
        metrics_utils.RestAPICallCounterChecker(
            pytest.metrics_client,
            QUERY_ENDPOINT,
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
        response = pytest.client.post(
            QUERY_ENDPOINT,
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.unprocessable_entity
        response_utils.check_content_type(response, "application/json")


@pytest.mark.skip_with_lcore
def test_token_counters_for_query_call_with_improper_payload() -> None:
    """Check how the tokens counter are updated accordingly."""
    model, provider = metrics_utils.get_enabled_model_and_provider(
        pytest.metrics_client
    )

    with (
        metrics_utils.RestAPICallCounterChecker(
            pytest.metrics_client,
            QUERY_ENDPOINT,
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
        response = pytest.client.post(
            QUERY_ENDPOINT,
            json={"parameter": "this-is-not-proper-question-my-friend"},
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.unprocessable_entity
        response_utils.check_content_type(response, "application/json")


@pytest.mark.tool_calling
@pytest.mark.rag
@retry(max_attempts=3, wait_between_runs=10)
def test_rag_question() -> None:
    """Ensure responses include rag references."""
    with metrics_utils.RestAPICallCounterChecker(pytest.metrics_client, QUERY_ENDPOINT):
        response = pytest.client.post(
            QUERY_ENDPOINT,
            json={"query": "about openshift virtualization"},
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.ok
        response_utils.check_content_type(response, "application/json")

        print(vars(response))
        json_response = response.json()
        assert "conversation_id" in json_response
        assert len(json_response["referenced_documents"]) > 2
        assert "virt" in json_response["referenced_documents"][0]["doc_url"]
        assert "https://" in json_response["referenced_documents"][0]["doc_url"]
        assert json_response["referenced_documents"][0]["doc_title"]

        # Length should be same, as there won't be duplicate entry
        doc_urls_list = [rd["doc_url"] for rd in json_response["referenced_documents"]]
        assert len(doc_urls_list) == len(set(doc_urls_list))


@pytest.mark.skip_with_lcore
@pytest.mark.cluster
def test_query_filter() -> None:
    """Ensure responses does not include filtered words and redacted words are not logged."""
    with metrics_utils.RestAPICallCounterChecker(pytest.metrics_client, QUERY_ENDPOINT):
        query = "what is foo in bar?"
        response = pytest.client.post(
            QUERY_ENDPOINT,
            json={"query": query},
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.ok
        response_utils.check_content_type(response, "application/json")
        print(vars(response))
        json_response = response.json()
        assert "conversation_id" in json_response
        # values to be filtered and replaced are defined in:
        # tests/config/singleprovider.e2e.template.config.yaml
        response_text = json_response["response"].lower()
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
                assert pattern not in line.lower(), f"failed for {pattern}"

        # Ensure the intended redaction has occurred
        assert "what is deployment in openshift?" in container_log


@retry(max_attempts=3, wait_between_runs=10)
def test_conversation_history() -> None:
    """Ensure conversations include previous query history."""
    with metrics_utils.RestAPICallCounterChecker(pytest.metrics_client, QUERY_ENDPOINT):
        response = pytest.client.post(
            QUERY_ENDPOINT,
            json={
                "query": "what is ingress in kubernetes?",
            },
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )
        debug_msg = "First call to LLM without conversation history has failed"
        assert response.status_code == requests.codes.ok, debug_msg
        response_utils.check_content_type(response, "application/json", debug_msg)

        print(vars(response))
        json_response = response.json()
        response_text = json_response["response"].lower()
        assert "ingress" in response_text, debug_msg

        # get the conversation id so we can reuse it for the follow up question
        cid = json_response["conversation_id"]
        response = pytest.client.post(
            QUERY_ENDPOINT,
            json={"conversation_id": cid, "query": "what?"},
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )
        print(vars(response))

        debug_msg = "Second call to LLM with conversation history has failed"
        assert response.status_code == requests.codes.ok
        response_utils.check_content_type(response, "application/json", debug_msg)

        json_response = response.json()
        response_text = json_response["response"].lower()
        assert "ingress" in response_text, debug_msg


@pytest.mark.skip_with_lcore
def test_query_with_provider_but_not_model() -> None:
    """Check the REST API /v1/query with POST HTTP method for provider specified, but no model."""
    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client,
        QUERY_ENDPOINT,
        status_code=requests.codes.unprocessable_entity,
    ):
        # just the provider is explicitly specified, but model selection is missing
        response = pytest.client.post(
            QUERY_ENDPOINT,
            json={
                "conversation_id": "",
                "query": "what is kubernetes?",
                "provider": "bam",
            },
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.unprocessable_entity
        response_utils.check_content_type(response, "application/json")

        json_response = response.json()

        # error thrown on Pydantic level
        assert (
            json_response["detail"][0]["msg"]
            == "Value error, LLM model must be specified when the provider is specified."
        )


@pytest.mark.skip_with_lcore
def test_query_with_model_but_not_provider() -> None:
    """Check the REST API /v1/query with POST HTTP method for model specified, but no provider."""
    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client,
        QUERY_ENDPOINT,
        status_code=requests.codes.unprocessable_entity,
    ):
        # just model is explicitly specified, but provider selection is missing
        response = pytest.client.post(
            QUERY_ENDPOINT,
            json={
                "conversation_id": "",
                "query": "what is kubernetes?",
                "model": "model-name",
            },
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.unprocessable_entity
        response_utils.check_content_type(response, "application/json")

        json_response = response.json()

        assert (
            json_response["detail"][0]["msg"]
            == "Value error, LLM provider must be specified when the model is specified."
        )


@pytest.mark.skip_with_lcore
def test_query_with_unknown_provider() -> None:
    """Check the REST API /v1/query with POST HTTP method for unknown provider specified."""
    # retrieve currently selected model
    model, _ = metrics_utils.get_enabled_model_and_provider(pytest.metrics_client)

    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client,
        QUERY_ENDPOINT,
        status_code=requests.codes.unprocessable_entity,
    ):
        # provider is unknown
        response = pytest.client.post(
            QUERY_ENDPOINT,
            json={
                "conversation_id": "",
                "query": "what is kubernetes?",
                "provider": "foo",
                "model": model,
            },
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.unprocessable_entity
        response_utils.check_content_type(response, "application/json")

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


@pytest.mark.skip_with_lcore
def test_query_with_unknown_model() -> None:
    """Check the REST API /v1/query with POST HTTP method for unknown model specified."""
    # retrieve currently selected provider
    _, provider = metrics_utils.get_enabled_model_and_provider(pytest.metrics_client)

    with metrics_utils.RestAPICallCounterChecker(
        pytest.metrics_client,
        QUERY_ENDPOINT,
        status_code=requests.codes.unprocessable_entity,
    ):
        # model is unknown
        response = pytest.client.post(
            QUERY_ENDPOINT,
            json={
                "conversation_id": "",
                "query": "what is kubernetes?",
                "provider": provider,
                "model": "bar",
            },
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.unprocessable_entity
        response_utils.check_content_type(response, "application/json")

        json_response = response.json()

        # explicit response and cause check
        assert (
            "detail" in json_response
        ), "Improper response format: 'detail' node is missing"
        assert "Unable to process this request" in json_response["detail"]["response"]
        assert "Model 'bar' is not a valid model " in json_response["detail"]["cause"]


@pytest.mark.tool_calling
@retry(max_attempts=3, wait_between_runs=10)
def test_tool_calling() -> None:
    """Check the REST API /v1/query with POST HTTP method for tool calling."""
    with metrics_utils.RestAPICallCounterChecker(pytest.metrics_client, QUERY_ENDPOINT):
        cid = suid.get_suid()
        response = pytest.client.post(
            QUERY_ENDPOINT,
            json={
                "conversation_id": cid,
                "query": "What pods are currently running in the openshift-lightspeed namespace?",
            },
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.ok

        response_utils.check_content_type(response, "application/json")
        print(vars(response))
        json_response = response.json()

        # checking a few major information from response
        assert json_response["conversation_id"] == cid

        # Sometime granite doesn't summarize well,
        # response may contain actual tool commands.
        assert re.search(
            r"(lightspeed-app-server|\[\"pods\", \"-n\", \"openshift-lightspeed\"\])",
            json_response["response"],
        )
        assert json_response["input_tokens"] > 0
        assert json_response["output_tokens"] > 0

        # Special check for granite
        assert not json_response["response"].strip().startswith("<tool_call>")


@pytest.mark.byok1
def test_rag_question_byok1() -> None:
    """Ensure response include expected top rag reference."""
    with metrics_utils.RestAPICallCounterChecker(pytest.metrics_client, QUERY_ENDPOINT):
        response = pytest.client.post(
            QUERY_ENDPOINT,
            json={"query": "about openshift virtualization"},
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.ok

        print(vars(response))
        assert "4.17" in response.json()["referenced_documents"][0]["doc_url"]


@pytest.mark.byok2
def test_rag_question_byok2() -> None:
    """Ensure response include expected top rag reference."""
    with metrics_utils.RestAPICallCounterChecker(pytest.metrics_client, QUERY_ENDPOINT):
        response = pytest.client.post(
            QUERY_ENDPOINT,
            json={"query": "about openshift virtualization"},
            timeout=test_api.LLM_REST_API_TIMEOUT,
        )
        assert response.status_code == requests.codes.ok

        print(vars(response))
        assert "4.16" in response.json()["referenced_documents"][0]["doc_url"]
