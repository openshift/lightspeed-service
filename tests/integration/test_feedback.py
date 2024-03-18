"""Integration tests for REST API endpoints for providing user feedback."""

import os
from unittest.mock import patch

import pytest
import requests
from fastapi.testclient import TestClient

from ols.app.models.config import UserDataCollection
from ols.utils import config, suid
from ols.utils.suid import check_suid


# we need to patch the config file path to point to the test
# config file before we import anything from main.py
@pytest.fixture(scope="module")
@patch.dict(os.environ, {"OLS_CONFIG_FILE": "tests/config/valid_config.yaml"})
def setup():
    """Setups the test client."""
    global client
    config.init_config("tests/config/valid_config.yaml")
    from ols.app.main import app

    client = TestClient(app)
    config.dev_config.disable_auth = True


@pytest.fixture
def with_disabled_feedback(tmpdir):
    """Fixture disables feedback."""
    config.ols_config.user_data_collection = UserDataCollection(feedback_disabled=True)
    yield


@pytest.fixture
def with_enabled_feedback(tmpdir):
    """Fixture enables feedback and configures its location."""
    config.ols_config.user_data_collection = UserDataCollection(
        feedback_disabled=False, feedback_storage=tmpdir.strpath
    )
    yield


def test_feedback_endpoints_disabled_when_set_in_config(with_disabled_feedback):
    """Check if feedback endpoints are disabled when set in config."""
    # status endpoint is always available
    response = client.get("/v1/feedback/status")
    assert response.status_code == requests.codes.ok

    response = client.get("/v1/feedback/list")
    assert response.status_code == requests.codes.forbidden

    response = client.post("/v1/feedback/", json={"a": 5})
    assert response.status_code == requests.codes.forbidden

    response = client.delete("/v1/feedback/12345678-abcd-0000-0123-456789abcdef")
    assert response.status_code == requests.codes.forbidden


def test_feedback_status(with_enabled_feedback):
    """Check if feedback status is returned."""
    response = client.get("/v1/feedback/status")
    assert response.status_code == requests.codes.ok
    assert response.json() == {"functionality": "feedback", "status": {"enabled": True}}


def test_feedback(with_enabled_feedback):
    """Check if feedback with correct format is accepted by the service."""
    # use proper conversation ID
    conversation_id = suid.get_suid()

    response = client.post(
        "/v1/feedback",
        json={
            "conversation_id": conversation_id,
            "user_question": "what are you doing?",
            "llm_response": "I don't know",
            "sentiment": -1,
        },
    )
    assert response.status_code == requests.codes.ok
    assert response.json() == {"response": "feedback received"}


def test_feedback_wrong_request(with_enabled_feedback):
    """Check if feedback with wrong payload (empty one) is not accepted by the service."""
    response = client.post("/v1/feedback", json={})
    # for the request send w/o proper payload, the server
    # should respond with proper error code
    assert response.status_code == requests.codes.unprocessable_entity


def test_feedback_mandatory_fields_not_provided_filled_in_request(
    with_enabled_feedback,
):
    """Check if feedback without mandatory fields is not accepted by the service."""
    # use proper conversation ID
    conversation_id = suid.get_suid()

    response = client.post(
        "/v1/feedback",
        json={
            "conversation_id": conversation_id,
            "user_question": "what are you doing?",
            "llm_response": "I don't know",
        },
    )
    # for the request send w/o proper payload, the server
    # should respond with proper error code
    assert response.status_code == requests.codes.unprocessable_entity


def test_feedback_no_payload_send(with_enabled_feedback):
    """Check if feedback without feedback payload."""
    response = client.post("/v1/feedback")
    # for the request send w/o payload, the server
    # should respond with proper error code
    assert response.status_code == requests.codes.unprocessable_entity


def test_feedback_list(with_enabled_feedback):
    """Check if feedback list is returned."""
    # store some feedback first
    conversation_id = suid.get_suid()
    response = client.post(
        "/v1/feedback",
        json={
            "conversation_id": conversation_id,
            "user_question": "what are you doing?",
            "llm_response": "I don't know",
            "sentiment": -1,
        },
    )
    assert response.status_code == requests.codes.ok

    response = client.get("/v1/feedback/list")

    assert response.status_code == requests.codes.ok
    assert len(response.json()["feedbacks"]) == 1
    assert check_suid(response.json()["feedbacks"][0])


def test_feedback_remove(with_enabled_feedback):
    """Check if feedback is removed."""
    # store some feedback first
    conversation_id = suid.get_suid()
    with patch("ols.app.endpoints.feedback.get_suid", return_value=conversation_id):
        response = client.post(
            "/v1/feedback",
            json={
                "conversation_id": conversation_id,
                "user_question": "what are you doing?",
                "llm_response": "I don't know",
                "sentiment": -1,
            },
        )
    assert response.status_code == requests.codes.ok

    response = client.delete(f"/v1/feedback/{conversation_id}")

    assert response.status_code == requests.codes.ok
    assert response.json() == {"response": "feedback removed"}
