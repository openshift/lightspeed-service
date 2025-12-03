"""End to end tests for the REST API endpoint /feedback."""

# we add new attributes into pytest instance, which is not recognized
# properly by linters
# pyright: reportAttributeAccessIssue=false

import pytest
import requests

from tests.e2e.utils import cluster as cluster_utils
from tests.e2e.utils import response as response_utils

from . import test_api


@pytest.mark.cluster
def test_feedback_can_post_with_wrong_token():
    """Test posting feedback with improper auth. token."""
    response = pytest.client.post(
        "/v1/feedback",
        json={
            "conversation_id": test_api.CONVERSATION_ID,
            "user_question": "what is OCP4?",
            "llm_response": "Openshift 4 is ...",
            "sentiment": 1,
        },
        timeout=test_api.BASIC_ENDPOINTS_TIMEOUT,
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert response.status_code == requests.codes.forbidden


@pytest.mark.data_export
def test_feedback_storing_cluster():
    """Test if the feedbacks are stored properly."""
    feedbacks_path = test_api.OLS_USER_DATA_PATH + "/feedback"
    pod_name = cluster_utils.get_pod_by_prefix()[0]

    # there are multiple tests running agains cluster, so transcripts
    # can be already present - we need to ensure the storage is empty
    # for this test
    feedbacks = cluster_utils.list_path(pod_name, feedbacks_path)
    if feedbacks:
        cluster_utils.remove_dir(pod_name, feedbacks_path)
        assert cluster_utils.list_path(pod_name, feedbacks_path) is None

    response = pytest.client.post(
        "/v1/feedback",
        json={
            "conversation_id": test_api.CONVERSATION_ID,
            "user_question": "what is OCP4?",
            "llm_response": "Openshift 4 is ...",
            "sentiment": 1,
        },
        timeout=test_api.BASIC_ENDPOINTS_TIMEOUT,
    )

    assert response.status_code == requests.codes.ok

    feedback_data = cluster_utils.get_single_existing_feedback(pod_name, feedbacks_path)

    assert feedback_data["user_id"]  # we don't care about actual value
    assert feedback_data["conversation_id"] == test_api.CONVERSATION_ID
    assert feedback_data["user_question"] == "what is OCP4?"
    assert feedback_data["llm_response"] == "Openshift 4 is ..."
    assert feedback_data["sentiment"] == 1


@pytest.mark.data_export
def test_feedback_missing_conversation_id():
    """Test posting feedback with missing conversation ID."""
    response = pytest.client.post(
        "/v1/feedback",
        json={
            "user_question": "what is OCP4?",
            "llm_response": "Openshift 4 is ...",
            "sentiment": 1,
        },
        timeout=test_api.BASIC_ENDPOINTS_TIMEOUT,
    )

    response_utils.check_missing_field_response(response, "conversation_id")


@pytest.mark.data_export
def test_feedback_missing_user_question():
    """Test posting feedback with missing user question."""
    response = pytest.client.post(
        "/v1/feedback",
        json={
            "conversation_id": test_api.CONVERSATION_ID,
            "llm_response": "Openshift 4 is ...",
            "sentiment": 1,
        },
        timeout=test_api.BASIC_ENDPOINTS_TIMEOUT,
    )

    response_utils.check_missing_field_response(response, "user_question")


@pytest.mark.data_export
def test_feedback_missing_llm_response():
    """Test posting feedback with missing LLM response."""
    response = pytest.client.post(
        "/v1/feedback",
        json={
            "conversation_id": test_api.CONVERSATION_ID,
            "user_question": "what is OCP4?",
            "sentiment": 1,
        },
        timeout=test_api.BASIC_ENDPOINTS_TIMEOUT,
    )

    response_utils.check_missing_field_response(response, "llm_response")


@pytest.mark.data_export
def test_feedback_improper_conversation_id():
    """Test posting feedback with improper conversation ID."""
    response = pytest.client.post(
        "/v1/feedback",
        json={
            "conversation_id": "incorrect-conversation-id",
            "user_question": "what is OCP4?",
            "llm_response": "Openshift 4 is ...",
            "sentiment": 1,
        },
        timeout=test_api.BASIC_ENDPOINTS_TIMEOUT,
    )

    # error should be detected on Pydantic level
    assert response.status_code == requests.codes.unprocessable

    # for incorrect conversation ID, the payload should be valid JSON
    response_utils.check_content_type(response, "application/json")
    json_response = response.json()

    assert (
        "detail" in json_response
    ), "Improper response format: 'detail' node is missing"
    assert (
        json_response["detail"][0]["msg"]
        == "Value error, Improper conversation ID incorrect-conversation-id"
    )
