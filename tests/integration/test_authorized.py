"""Integration tests for /livenss and /readiness REST API endpoints."""

from unittest.mock import patch

import pytest
import requests
from fastapi.testclient import TestClient

from ols import config, constants
from tests.mock_classes.mock_k8s_api import (
    mock_subject_access_review_response,
    mock_token_review_response,
)


@pytest.fixture(scope="function")
def _setup():
    """Setups the test client."""
    global client
    config.reload_from_yaml_file("tests/config/valid_config.yaml")

    # app.main need to be imported after the configuration is read
    from ols.app.main import app

    client = TestClient(app)


@pytest.fixture
def _disabled_auth():
    """Fixture for tests that expect disabled auth."""
    assert config.dev_config is not None
    config.dev_config.disable_auth = True


@pytest.fixture
def _enabled_auth():
    """Fixture for tests that expect enabled auth."""
    assert config.dev_config is not None
    config.dev_config.disable_auth = False


def test_post_authorized_disabled(_setup, _disabled_auth):
    """Check the REST API /v1/query with POST HTTP method when no payload is posted."""
    # perform POST request with authentication disabled
    response = client.post("/authorized")
    assert response.status_code == requests.codes.ok

    # check the response payload
    assert response.json() == {
        "user_id": constants.DEFAULT_USER_UID,
        "username": constants.DEFAULT_USER_NAME,
    }


def test_post_authorized_no_token(_setup, _enabled_auth):
    """Check the REST API /v1/query with POST HTTP method when no payload is posted."""
    # perform POST request without any payload
    response = client.post("/authorized")
    assert response.status_code == 401


@patch("ols.utils.auth_dependency.K8sClientSingleton.get_authn_api")
@patch("ols.utils.auth_dependency.K8sClientSingleton.get_authz_api")
def test_is_user_authorized_valid_token(
    mock_authz_api, mock_authn_api, _setup, _enabled_auth
):
    """Tests the is_user_authorized function with a mocked valid-token."""
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
def test_is_user_authorized_invalid_token(
    mock_authz_api, mock_authn_api, _setup, _enabled_auth
):
    """Test the is_user_authorized function with a mocked invalid-token."""
    # Setup mock responses for invalid token
    mock_authn_api.return_value.create_token_review.side_effect = (
        mock_token_review_response
    )
    mock_authz_api.return_value.create_subject_access_review.side_effect = (
        mock_subject_access_review_response
    )

    response = client.post(
        "/authorized",
        headers=[(b"authorization", b"Bearer invalid-token")],
    )
    assert response.status_code == 403
