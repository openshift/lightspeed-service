"""Integration tests for basic OLS REST API endpoints."""

import json
import os
import pickle
import re
import sys
import time
from typing import Optional

import pytest
import requests
from httpx import Client

from ols.constants import (
    HTTP_REQUEST_HEADERS_TO_REDACT,
    INVALID_QUERY_RESP,
)
from ols.utils import suid
from tests.e2e import (
    cluster_utils,
    helper_utils,
    metrics_utils,
)
from tests.e2e.constants import (
    BASIC_ENDPOINTS_TIMEOUT,
    CONVERSATION_ID,
    EVAL_THRESHOLD,
    LLM_REST_API_TIMEOUT,
    NON_LLM_REST_API_TIMEOUT,
)
from tests.scripts.must_gather import must_gather
from tests.scripts.validate_response import ResponseValidation

from .postgres_utils import (
    read_conversation_history,
    read_conversation_history_count,
    retrieve_connection,
)
from .test_decorators import retry

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
client: Client
metrics_client: Client


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

        client = helper_utils.get_http_client(ols_url, token)
        metrics_client = helper_utils.get_http_client(ols_url, metrics_token)
    except Exception as e:
        print(f"Failed to setup ols access: {e}")
        sys.exit(1)


def teardown_module(module):
    """Clean up the environment after all tests are executed."""
    if on_cluster:
        must_gather()


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


def test_repeatedly():
    for i in range(1000):
        test_openapi_endpoint()
