"""Integration tests for basic OLS REST API endpoints."""

# we add new attributes into pytest instance, which is not recognized
# properly by linters
# pyright: reportAttributeAccessIssue=false

import json
import re
import time

import pytest
import requests

from ols.constants import HTTP_REQUEST_HEADERS_TO_REDACT
from ols.customize import metadata
from ols.utils import suid
from tests.e2e.utils import client as client_utils
from tests.e2e.utils import cluster as cluster_utils
from tests.e2e.utils import metrics as metrics_utils
from tests.e2e.utils import response as response_utils
from tests.e2e.utils.constants import (
    BASIC_ENDPOINTS_TIMEOUT,
    CONVERSATION_ID,
    LLM_REST_API_TIMEOUT,
    NON_LLM_REST_API_TIMEOUT,
    OLS_COLLECTOR_DISABLING_FILE,
    OLS_USER_DATA_COLLECTION_INTERVAL,
    OLS_USER_DATA_PATH,
)
from tests.e2e.utils.decorators import retry
from tests.e2e.utils.postgres import (
    read_conversation_history,
    read_conversation_history_count,
    retrieve_connection,
)


@pytest.fixture(name="postgres_connection", scope="module")
def fixture_postgres_connection():
    """Fixture with Postgres connection."""
    return retrieve_connection()


@pytest.mark.smoketest
@retry(max_attempts=3, wait_between_runs=10)
def test_readiness():
    """Test handler for /readiness REST API endpoint."""
    endpoint = "/readiness"
    with metrics_utils.RestAPICallCounterChecker(pytest.metrics_client, endpoint):
        response = pytest.client.get(endpoint, timeout=LLM_REST_API_TIMEOUT)
        assert response.status_code == requests.codes.ok
        response_utils.check_content_type(response, "application/json")
        assert response.json() == {"ready": True, "reason": "service is ready"}


@pytest.mark.smoketest
def test_liveness():
    """Test handler for /liveness REST API endpoint."""
    endpoint = "/liveness"
    with metrics_utils.RestAPICallCounterChecker(pytest.metrics_client, endpoint):
        response = pytest.client.get(endpoint, timeout=BASIC_ENDPOINTS_TIMEOUT)
        assert response.status_code == requests.codes.ok
        response_utils.check_content_type(response, "application/json")
        assert response.json() == {"alive": True}


def test_metrics() -> None:
    """Check if service provides metrics endpoint with expected metrics."""
    response = pytest.metrics_client.get("/metrics", timeout=BASIC_ENDPOINTS_TIMEOUT)
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
    model, provider = metrics_utils.get_enabled_model_and_provider(
        pytest.metrics_client
    )

    # enabled model must be one of our expected combinations
    assert model, provider in {
        ("gpt-4o-mini", "openai"),
        ("gpt-4o-mini", "azure_openai"),
        ("ibm/granite-3-2-8b-instruct", "watsonx"),
    }


def test_one_default_model_provider():
    """Check if one model and provider is selected as default."""
    states = metrics_utils.get_enable_status_for_all_models(pytest.metrics_client)
    enabled_states = [state for state in states if state is True]
    assert (
        len(enabled_states) == 1
    ), "one model and provider should be selected as default"


@pytest.mark.cluster
def test_improper_token():
    """Test accessing /v1/query endpoint using improper auth. token."""
    response = pytest.client.post(
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
    response = pytest.metrics_client.post(
        "/v1/query",
        json={"query": "what is foo in bar?"},
        timeout=NON_LLM_REST_API_TIMEOUT,
    )
    assert response.status_code == requests.codes.forbidden
    response = pytest.client.get("/metrics", timeout=BASIC_ENDPOINTS_TIMEOUT)
    assert response.status_code == requests.codes.forbidden


@pytest.mark.cluster
def test_transcripts_storing_cluster():
    """Test if the transcripts are stored properly."""
    transcripts_path = OLS_USER_DATA_PATH + "/transcripts"
    cluster_utils.wait_for_running_pod()
    pod_name = cluster_utils.get_pod_by_prefix()[0]
    # disable collector script to avoid interference with the test
    cluster_utils.create_file(pod_name, OLS_COLLECTOR_DISABLING_FILE, "")

    # there are multiple tests running agains cluster, so transcripts
    # can be already present - we need to ensure the storage is empty
    # for this test
    transcripts = cluster_utils.list_path(pod_name, transcripts_path)
    if transcripts:
        cluster_utils.remove_dir(pod_name, transcripts_path)
        assert cluster_utils.list_path(pod_name, transcripts_path) is None

    response = pytest.client.post(
        "/v1/query",
        json={
            "query": "what is kubernetes?",
            "attachments": [
                {
                    "attachment_type": "log",
                    "content_type": "text/plain",
                    # Sample content
                    "content": "Kubernetes is a core component of OpenShift.",
                }
            ],
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
    assert "rag_chunks" in transcript

    assert transcript["query_is_valid"]
    assert isinstance(transcript["rag_chunks"], list)
    assert len(transcript["rag_chunks"])
    assert transcript["rag_chunks"][0]["text"]
    assert transcript["rag_chunks"][0]["doc_url"]
    assert transcript["rag_chunks"][0]["doc_title"]
    assert "truncated" in transcript

    # check the attachment node existence and its content
    assert "attachments" in transcript

    expected_attachment_node = [
        {
            "attachment_type": "log",
            "content_type": "text/plain",
            "content": "Kubernetes is a core component of OpenShift.",
        }
    ]
    assert transcript["attachments"] == expected_attachment_node
    assert transcript["tool_calls"] == []


@retry(max_attempts=3, wait_between_runs=10)
def test_openapi_endpoint():
    """Test handler for /opanapi REST API endpoint."""
    response = pytest.client.get("/openapi.json", timeout=BASIC_ENDPOINTS_TIMEOUT)
    assert response.status_code == requests.codes.ok
    response_utils.check_content_type(response, "application/json")

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

    # retrieve pre-generated OpenAPI schema
    with open("docs/openapi.json", encoding="utf-8") as fin:
        expected_schema = json.load(fin)

    # remove node that is not included in pre-generated OpenAPI schema
    del payload["info"]["license"]

    # compare schemas (as dicts)
    assert (
        payload == expected_schema
    ), "OpenAPI schema returned from service does not have expected content."


def test_cache_existence(postgres_connection):
    """Test the cache existence."""
    if postgres_connection is None:
        pytest.skip("Postgres is not accessible.")

    value = read_conversation_history_count(postgres_connection)
    # check if history exists at all
    assert value is not None


def test_conversation_in_postgres_cache(postgres_connection) -> None:
    """Check how/if the conversation is stored in cache."""
    if postgres_connection is None:
        pytest.skip("Postgres is not accessible.")

    cid = suid.get_suid()
    client_utils.perform_query(pytest.client, cid, "what is kubernetes?")

    conversation, updated_at = read_conversation_history(postgres_connection, cid)
    assert conversation is not None
    assert updated_at is not None

    # deserialize conversation into list of messages
    deserialized = json.loads(conversation)
    assert deserialized is not None

    # we expect one question + one answer
    assert len(deserialized) == 2

    # question check
    assert "what is kubernetes?" in deserialized[0].content

    # trivial check for answer (exact check is done in different tests)
    assert "Kubernetes" in deserialized[1].content

    # second question
    client_utils.perform_query(pytest.client, cid, "what is openshift virtualization?")

    conversation, updated_at = read_conversation_history(postgres_connection, cid)
    assert conversation is not None

    # unpickle conversation into list of messages
    deserialized = json.loads(conversation, errors="strict")
    assert deserialized is not None

    # we expect one question + one answer
    assert len(deserialized) == 4

    # first question
    assert "what is kubernetes?" in deserialized[0].content

    # first answer
    assert "Kubernetes" in deserialized[1].content

    # second question
    assert "what is openshift virtualization?" in deserialized[2].content

    # second answer
    assert "OpenShift" in deserialized[3].content


@pytest.mark.cluster
def test_user_data_collection():
    """Test user data collection.

    It is performed by checking the user data collection container logs
    for the expected messages in logs.
    A bit of trick is required to check just the logs since the last
    action (as container logs can be influenced by other tests).
    """
    pod_name = None
    try:
        pod_name = cluster_utils.get_pod_by_prefix()[0]

        # enable collector script
        if pod_name is not None:
            cluster_utils.remove_file(pod_name, OLS_COLLECTOR_DISABLING_FILE)
            assert "disable_collector" not in cluster_utils.list_path(
                pod_name, OLS_USER_DATA_PATH
            )

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
        data_collection_container_name = "lightspeed-service-user-data-collector"
        pod_name = cluster_utils.get_pod_by_prefix()[0]

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
        time.sleep(OLS_USER_DATA_COLLECTION_INTERVAL + 5)

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
        response = pytest.client.post(
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
        # ensure the script have enought time to send the payload before
        # we pull its logs
        time.sleep(OLS_USER_DATA_COLLECTION_INTERVAL + 5)

        # check that data was packaged, sent and removed
        container_log = cluster_utils.get_container_log(
            pod_name, data_collection_container_name
        )
        logs = filter_logs(container_log, last_log_line)
        assert "collected 1 files (splitted to 1 chunks) from" in logs
        assert "data uploaded with request_id:" in logs
        assert "uploaded data removed" in logs
        assert "data upload failed with response:" not in logs
        user_data = cluster_utils.list_path(pod_name, OLS_USER_DATA_PATH + "/feedback/")
        assert user_data == []

    finally:
        # disable collector script after test/on failure
        if pod_name is not None:
            cluster_utils.create_file(pod_name, OLS_COLLECTOR_DISABLING_FILE, "")


@pytest.mark.cluster
def test_http_header_redaction():
    """Test that sensitive HTTP headers are redacted from the logs."""
    for header in HTTP_REQUEST_HEADERS_TO_REDACT:
        endpoint = "/liveness"
        with metrics_utils.RestAPICallCounterChecker(pytest.metrics_client, endpoint):
            response = pytest.client.get(
                endpoint,
                headers={f"{header}": "some_value"},
                timeout=BASIC_ENDPOINTS_TIMEOUT,
            )
            assert response.status_code == requests.codes.ok
            response_utils.check_content_type(response, "application/json")
            assert response.json() == {"alive": True}

    container_log = cluster_utils.get_container_log(
        cluster_utils.get_pod_by_prefix()[0], "lightspeed-service-api"
    )

    for header in HTTP_REQUEST_HEADERS_TO_REDACT:
        assert f'"{header}":"XXXXX"' in container_log
        assert f'"{header}":"some_value"' not in container_log


@pytest.mark.azure_entra_id
def test_azure_entra_id():
    """Test single question via Azure Entra ID credentials."""
    response = pytest.client.post(
        "/v1/query",
        json={
            "query": "what is kubernetes?",
            "provider": "azure_openai_with_entra_id",
            "model": "gpt-4o-mini",
        },
        timeout=LLM_REST_API_TIMEOUT,
    )
    assert response.status_code == requests.codes.ok

    response_utils.check_content_type(response, "application/json")
    print(vars(response))
    json_response = response.json()

    # checking a few major information from response
    assert "Kubernetes is" in json_response["response"]
    assert re.search(
        r"orchestration (tool|system|platform|engine)",
        json_response["response"],
        re.IGNORECASE,
    )


@pytest.mark.certificates
def test_generated_service_certs_rotation():
    """Verify OLS responds after certificate rotation."""
    service_tls = cluster_utils.get_certificate_secret_name()
    cluster_utils.delete_resource(
        resource="secret", name=service_tls, namespace="openshift-lightspeed"
    )
    response = pytest.client.post(
        "/v1/query",
        json={"query": "what is kubernetes?"},
        timeout=LLM_REST_API_TIMEOUT,
    )
    assert response.status_code == requests.codes.ok


@pytest.mark.certificates
def test_ca_service_certs_rotation():
    """Verify OLS responds after ca certificate rotation."""
    cluster_utils.delete_resource(
        resource="secret", name="signing-key", namespace="openshift-service-ca"
    )
    response = pytest.client.post(
        "/v1/query",
        json={"query": "what is kubernetes?"},
        timeout=LLM_REST_API_TIMEOUT,
    )
    assert response.status_code == requests.codes.ok
    cluster_utils.restart_deployment(
        name="lightspeed-operator-controller-manager", namespace="openshift-lightspeed"
    )
    cluster_utils.restart_deployment(
        name="lightspeed-app-server", namespace="openshift-lightspeed"
    )
    cluster_utils.restart_deployment(
        name="lightspeed-console-plugin", namespace="openshift-lightspeed"
    )
    # Wait for service to become available again
    time.sleep(120)
    cluster_utils.wait_for_running_pod()

    response = pytest.client.post(
        "/v1/query",
        json={"query": "what is kubernetes?"},
        timeout=LLM_REST_API_TIMEOUT,
    )
    assert response.status_code == requests.codes.ok


@pytest.mark.quota_limits
def test_quota_limits():
    """Verify OLS quota limits."""
    _, provider = metrics_utils.get_enabled_model_and_provider(pytest.metrics_client)
    response = pytest.client.post(
                "/v1/query",
                json={"query": "what is kubernetes?"},
                timeout=LLM_REST_API_TIMEOUT,
            )

    # assert that the available quota is
    # less than the initial one hardcoded in the olsconfig
    assert (
        response.json()["available_quotas"]["UserQuotaLimiter"] < 11111
    ), "available quota was not used or user limits were ignored"
    assert (
        response.json()["input_tokens"] > 0
    ), "input tokens were not populated"
    # Remove the user limitation, wait for pod and check available quota again
    action = {"op": "remove", "path": "/spec/ols/quotaHandlersConfig/limitersConfig/1"}
    merge_strat = f"-p=[{action}]"
    cluster_utils.run_oc(["patch", "olsconfig", "cluster", "--type=json", merge_strat])
    cluster_utils.wait_for_running_pod()
    response = pytest.client.post(
        "/v1/query",
        json={"query": "what is kubernetes?"},
        timeout=LLM_REST_API_TIMEOUT,
    )
    # assert that the available quota is less than the initial one hardcoded in the olsconfig
    # but higher than the user limit
    available_quota = response.json()["available_quotas"]["ClusterQuotaLimiter"]
    assert (
        11111 < available_quota < 22222
    ), "Quota still being user limited when limitation was removed"
    assert response.json()["input_tokens"] > 0, "input tokens were not populated"
    cluster_utils.run_oc(
        ["apply", "-f", f"tests/config/operator_install/olsconfig.crd.{provider}.yaml"]
    )
    cluster_utils.wait_for_running_pod()
    cluster_utils.run_oc(
        ["scale", "--replicas=0", "deployment/lightspeed-operator-controller-manager"]
    )
    response = pytest.client.post(
        "/v1/query",
        json={"query": "what is kubernetes?"},
        timeout=LLM_REST_API_TIMEOUT,
    )
    with pytest.raises(KeyError):
        assert not response.json()["available_quotas"][
            "UserQuotaLimiter"
        ], "available quota populated after being removed from config"
    with pytest.raises(KeyError):
        assert not response.json()["available_quotas"][
            "ClusterQuotaLimiter"
        ], "available quota populated after being removed from config"
    assert response.json()["input_tokens"] > 0, "input tokens were not populated"
