"""Integration tests for REST API endpoints for providing user feedback."""

import os
from unittest.mock import patch

import requests
from fastapi.testclient import TestClient

from ols.utils import config, suid


# we need to patch the config file path to point to the test
# config file before we import anything from main.py
@patch.dict(os.environ, {"OLS_CONFIG_FILE": "tests/config/valid_config.yaml"})
def setup():
    """Setups the test client."""
    global client
    config.init_config("tests/config/valid_config.yaml")
    from ols.app.main import app

    client = TestClient(app)
    config.dev_config.disable_auth = True


def test_feedback():
    """Check if feedback with correct format is accepted by the service."""
    # TODO: should we validate that the correct log messages are written?

    # use proper conversation ID
    conversation_id = suid.get_suid()

    response = client.post(
        "/v1/feedback",
        json={"conversation_id": conversation_id, "feedback_object": {"blah": "bloh"}},
    )
    assert response.status_code == requests.codes.ok
    assert response.json() == {"response": "feedback received"}


def test_feedback_wrong_request():
    """Check if feedback with wrong payload (empty one) is not accepted by the service."""
    response = client.post("/v1/feedback", json={})
    # for the request send w/o proper payload, the server
    # should respond with proper error code
    assert response.status_code == requests.codes.unprocessable_entity


def test_feedback_wrong_not_filled_in_request():
    """Check if feedback without feedback object is not accepted by the service."""
    # use proper conversation ID
    conversation_id = suid.get_suid()

    response = client.post(
        "/v1/feedback",
        json={"conversation_id": conversation_id, "feedback_object": None},
    )
    # for the request send w/o proper payload, the server
    # should respond with proper error code
    assert response.status_code == requests.codes.unprocessable_entity


def test_feedback_no_payload_send():
    """Check if feedback without feedback payload."""
    response = client.post("/v1/feedback")
    # for the request send w/o payload, the server
    # should respond with proper error code
    assert response.status_code == requests.codes.unprocessable_entity


def test_feedback_wrong_conversation_id():
    """Check if feedback with wrong conversation ID is not accepted by the service."""
    response = client.post(
        "/v1/feedback",
        json={"conversation_id": 0, "feedback_object": "blah"},
    )
    # for the request send w/o proper payload, the server
    # should respond with proper error code
    assert response.status_code == requests.codes.unprocessable_entity
