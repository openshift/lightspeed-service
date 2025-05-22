"""Integration tests for REST API endpoints for providing user feedback."""

# we add new attributes into pytest instance, which is not recognized
# properly by linters
# pyright: reportAttributeAccessIssue=false

from unittest.mock import patch

import pytest
import requests
from fastapi.testclient import TestClient

from ols import config
from ols.app.models.config import UserDataCollection
from ols.utils import suid

# use proper conversation ID
CONVERSATION_ID = suid.get_suid()


@pytest.fixture(scope="function", autouse=True)
def _setup():
    """Setups the test client."""
    config.reload_from_yaml_file("tests/config/config_for_integration_tests.yaml")

    # app.main need to be imported after the configuration is read
    from ols.app.main import app  # pylint: disable=C0415

    pytest.client = TestClient(app)
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


@pytest.mark.usefixtures("_with_disabled_feedback")
def test_feedback_endpoints_disabled_when_set_in_config():
    """Check if feedback endpoints are disabled when set in config."""
    # status endpoint is always available
    response = pytest.client.get("/v1/feedback/status")
    assert response.status_code == requests.codes.ok

    response = pytest.client.post("/v1/feedback/", json={"a": 5})
    assert response.status_code == requests.codes.forbidden


@pytest.mark.usefixtures("_with_enabled_feedback")
def test_feedback_status():
    """Check if feedback status is returned."""
    response = pytest.client.get("/v1/feedback/status")
    assert response.status_code == requests.codes.ok
    assert response.json() == {"functionality": "feedback", "status": {"enabled": True}}


@pytest.mark.usefixtures("_with_enabled_feedback")
def test_feedback():
    """Check if feedback with correct format is accepted by the service."""
    response = pytest.client.post(
        "/v1/feedback",
        json={
            "conversation_id": CONVERSATION_ID,
            "user_question": "what are you doing?",
            "llm_response": "I don't know",
            "sentiment": -1,
        },
    )
    assert response.status_code == requests.codes.ok
    assert response.json() == {"response": "feedback received"}


@pytest.mark.usefixtures("_with_enabled_feedback")
def test_feedback_improper_conversation_id():
    """Check if feedback with improper conversation ID is rejected."""
    response = pytest.client.post(
        "/v1/feedback",
        json={
            "conversation_id": "really-not-an-uuid",
            "user_question": "what are you doing?",
            "llm_response": "I don't know",
            "sentiment": -1,
        },
    )
    assert response.status_code == requests.codes.unprocessable_entity


@pytest.mark.usefixtures("_with_enabled_feedback")
def test_feedback_improper_sentiment():
    """Check if feedback with improper sentiment value is rejected."""
    response = pytest.client.post(
        "/v1/feedback",
        json={
            "conversation_id": CONVERSATION_ID,
            "user_question": "what are you doing?",
            "llm_response": "I don't know",
            "sentiment": -2,
        },
    )
    assert response.status_code == requests.codes.unprocessable_entity

    response = pytest.client.post(
        "/v1/feedback",
        json={
            "conversation_id": CONVERSATION_ID,
            "user_question": "what are you doing?",
            "llm_response": "I don't know",
            "sentiment": "foo",
        },
    )
    assert response.status_code == requests.codes.unprocessable_entity


@pytest.mark.usefixtures("_with_enabled_feedback")
def test_feedback_wrong_request():
    """Check if feedback with wrong payload (empty one) is not accepted by the service."""
    response = pytest.client.post("/v1/feedback", json={})
    # for the request send w/o proper payload, the server
    # should respond with proper error code
    assert response.status_code == requests.codes.unprocessable_entity


@pytest.mark.usefixtures("_with_enabled_feedback")
def test_feedback_mandatory_fields_not_provided_filled_in_request():
    """Check if feedback without mandatory fields is not accepted by the service."""
    response = pytest.client.post(
        "/v1/feedback",
        json={
            "conversation_id": CONVERSATION_ID,
            "user_question": "what are you doing?",
            "llm_response": "I don't know",
        },
    )
    # for the request send w/o proper payload, the server
    # should respond with proper error code
    assert response.status_code == requests.codes.unprocessable_entity


@pytest.mark.usefixtures("_with_enabled_feedback")
def test_feedback_no_payload_send():
    """Check if feedback without feedback payload."""
    response = pytest.client.post("/v1/feedback")
    # for the request send w/o payload, the server
    # should respond with proper error code
    assert response.status_code == requests.codes.unprocessable_entity


@pytest.mark.usefixtures("_with_enabled_feedback")
def test_feedback_error_raised():
    """Check if feedback endpoint raises an exception when storing feedback fails."""
    with patch(
        "ols.app.endpoints.feedback.store_feedback",
        side_effect=Exception("Test exception"),
    ):
        response = pytest.client.post(
            "/v1/feedback",
            json={
                "conversation_id": CONVERSATION_ID,
                "user_question": "what are you doing?",
                "llm_response": "I don't know",
                "sentiment": -1,
            },
        )
        assert response.status_code == requests.codes.internal_server_error
