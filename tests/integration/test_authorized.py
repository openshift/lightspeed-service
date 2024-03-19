"""Integration tests for /livenss and /readiness REST API endpoints."""

import os
from unittest.mock import patch

import requests
from fastapi.testclient import TestClient

from ols import constants
from ols.utils import config
from tests.mock_classes.mock_k8s_api import (
    mock_subject_access_review_response,
    mock_token_review_response,
)


# we need to patch the config file path to point to the test
# config file before we import anything from main.py
@patch.dict(os.environ, {"OLS_CONFIG_FILE": "tests/config/valid_config.yaml"})
def setup():
    """Setups the test client."""
    global client
    config.init_config("tests/config/valid_config.yaml")
    from ols.app.main import app

    client = TestClient(app)


def test_post_authorized_disabled():
    """Check the REST API /v1/query with POST HTTP method when no payload is posted."""
    # perform POST request with authentication disabled
    config.dev_config.disable_auth = True
    response = client.post("/authorized")
    assert response.status_code == requests.codes.ok

    # check the response payload
    assert response.json() == {
        "user_id": constants.DEFAULT_USER_UID,
        "username": constants.DEFAULT_USER_NAME,
    }


def test_post_authorized_no_token():
    """Check the REST API /v1/query with POST HTTP method when no payload is posted."""
    # perform POST request without any payload
    config.dev_config.disable_auth = False
    response = client.post("/authorized")
    assert response.status_code == 401


@patch("ols.utils.auth_dependency.K8sClientSingleton.get_authn_api")
@patch("ols.utils.auth_dependency.K8sClientSingleton.get_authz_api")
def test_is_user_authorized_valid_token(mock_authz_api, mock_authn_api):
    """Tests the is_user_authorized function with a mocked valid-token."""
    config.dev_config.disable_auth = False
    # Setup mock responses for valid token
    mock_authn_api.return_value.create_token_review.side_effect = (
        mock_token_review_response
    )
    mock_authz_api.return_value.create_subject_access_review.side_effect = (
        mock_subject_access_review_response
    )
    response = client.post(
        "/authorized",
        headers=[(b"authorization", b"Bearer valid-token")],
    )
    assert response.status_code == requests.codes.ok
    print(response.json())

    # check the response payload
    assert response.json() == {"user_id": "valid-uid", "username": "valid-user"}


@patch("ols.utils.auth_dependency.K8sClientSingleton.get_authn_api")
@patch("ols.utils.auth_dependency.K8sClientSingleton.get_authz_api")
def test_is_user_authorized_invalid_token(mock_authz_api, mock_authn_api):
    """Test the is_user_authorized function with a mocked invalid-token."""
    config.dev_config.disable_auth = False
    # Setup mock responses for invalid token
    mock_authn_api.return_value.create_token_review.side_effect = (
        mock_token_review_response
    )
    mock_authz_api.return_value.create_subject_access_review.side_effect = (
        mock_subject_access_review_response
    )

    config.dev_config.disable_auth = False
    response = client.post(
        "/authorized",
        headers=[(b"authorization", b"Bearer invalid-token")],
    )
    assert response.status_code == 403
