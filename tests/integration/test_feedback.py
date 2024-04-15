"""Integration tests for REST API endpoints for providing user feedback."""

import pytest
import requests
from fastapi.testclient import TestClient

from ols.app.models.config import UserDataCollection
from ols.utils import config, suid


@pytest.fixture(scope="module", autouse=True)
def _setup():
    """Setups the test client."""
    global client
    config.init_config("tests/config/valid_config.yaml")

    # app.main need to be imported after the configuration is read
    from ols.app.main import app

    client = TestClient(app)
    config.dev_config.disable_auth = True


@pytest.fixture
def _with_disabled_feedback(tmpdir):
    """Fixture disables feedback."""
    config.ols_config.user_data_collection = UserDataCollection(feedback_disabled=True)


@pytest.fixture
def _with_enabled_feedback(tmpdir):
    """Fixture enables feedback and configures its location."""
    config.ols_config.user_data_collection = UserDataCollection(
        feedback_disabled=False, feedback_storage=tmpdir.strpath
    )


def test_feedback_endpoints_disabled_when_set_in_config(
    _setup, _with_disabled_feedback
):
    """Check if feedback endpoints are disabled when set in config."""
    # status endpoint is always available
    response = client.get("/v1/feedback/status")
    assert response.status_code == requests.codes.ok

    response = client.post("/v1/feedback/", json={"a": 5})
    assert response.status_code == requests.codes.forbidden


def test_feedback_status(_with_enabled_feedback):
    """Check if feedback status is returned."""
    response = client.get("/v1/feedback/status")
    assert response.status_code == requests.codes.ok
    assert response.json() == {"functionality": "feedback", "status": {"enabled": True}}


def test_feedback(_with_enabled_feedback):
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


def test_feedback_wrong_request(_with_enabled_feedback):
    """Check if feedback with wrong payload (empty one) is not accepted by the service."""
    response = client.post("/v1/feedback", json={})
    # for the request send w/o proper payload, the server
    # should respond with proper error code
    assert response.status_code == requests.codes.unprocessable_entity


def test_feedback_mandatory_fields_not_provided_filled_in_request(
    _with_enabled_feedback,
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


def test_feedback_no_payload_send(_with_enabled_feedback):
    """Check if feedback without feedback payload."""
    response = client.post("/v1/feedback")
    # for the request send w/o payload, the server
    # should respond with proper error code
    assert response.status_code == requests.codes.unprocessable_entity
