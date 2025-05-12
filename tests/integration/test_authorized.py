"""Integration tests for /livenss and /readiness REST API endpoints."""

import logging
from unittest.mock import patch

import pytest
import requests
from fastapi.testclient import TestClient

from ols import config, constants
from ols.app.models.config import LoggingConfig
from ols.utils.logging_configurator import configure_logging
from tests.mock_classes.mock_k8s_api import (
    mock_subject_access_review_response,
    mock_token_review_response,
)


@pytest.fixture(scope="function", autouse=True)
def _setup():
    """Setups the test client."""
    config.reload_from_yaml_file("tests/config/config_for_integration_tests.yaml")

    # app.main need to be imported after the configuration is read
    from ols.app.main import app  # pylint: disable=C0415

    pytest.client = TestClient(app)


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


@pytest.mark.usefixtures("_disabled_auth")
def test_post_authorized_disabled(caplog):
    """Check the REST API /v1/query with POST HTTP method with authentication disabled."""
    # perform POST request with authentication disabled
    logging_config = LoggingConfig(app_log_level="warning")

    configure_logging(logging_config)
    logger = logging.getLogger("ols")
    logger.handlers = [caplog.handler]  # add caplog handler to logger

    response = pytest.client.post("/authorized")
    assert response.status_code == requests.codes.ok

    # check the response payload
    assert response.json() == {
        "user_id": constants.DEFAULT_USER_UID,
        "username": constants.DEFAULT_USER_NAME,
        "skip_user_id_check": False,
    }

    # check if the auth checks warning message is found in the log
    captured_out = caplog.text
    assert "Auth checks disabled, skipping" in captured_out


@pytest.mark.usefixtures("_disabled_auth")
def test_post_authorized_disabled_with_logging_suppressed(caplog):
    """Check the REST API /v1/query with POST HTTP method with the auth warning suppressed."""
    # perform POST request with authentication disabled
    logging_config = LoggingConfig(app_log_level="warning")
    assert config.ols_config.logging_config is not None
    config.ols_config.logging_config.suppress_auth_checks_warning_in_log = True

    configure_logging(logging_config)
    logger = logging.getLogger("ols")
    logger.handlers = [caplog.handler]  # add caplog handler to logger

    response = pytest.client.post("/authorized")
    assert response.status_code == requests.codes.ok

    # check the response payload
    assert response.json() == {
        "user_id": constants.DEFAULT_USER_UID,
        "username": constants.DEFAULT_USER_NAME,
        "skip_user_id_check": False,
    }

    # check if the auth checks warning message is NOT found in the log
    captured_out = caplog.text
    assert "Auth checks disabled, skipping" not in captured_out


@pytest.mark.usefixtures("_enabled_auth")
def test_post_authorized_no_token():
    """Check the REST API /v1/query with POST HTTP method when no payload is posted."""
    # perform POST request without any payload
    response = pytest.client.post("/authorized")
    assert response.status_code == 401


@pytest.mark.usefixtures("_enabled_auth")
def test_is_user_authorized_valid_token():
    """Tests the is_user_authorized function with a mocked valid-token."""
    with (
        patch("ols.src.auth.k8s.K8sClientSingleton.get_authn_api") as mock_authn_api,
        patch("ols.src.auth.k8s.K8sClientSingleton.get_authz_api") as mock_authz_api,
    ):
        # Setup mock responses for valid token
        mock_authn_api.return_value.create_token_review.side_effect = (
            mock_token_review_response
        )
        mock_authz_api.return_value.create_subject_access_review.side_effect = (
            mock_subject_access_review_response
        )
        response = pytest.client.post(
            "/authorized",
            headers=[(b"authorization", b"Bearer valid-token")],
        )
        assert response.status_code == requests.codes.ok
        print(response.json())

        # check the response payload
        assert response.json() == {
            "user_id": "valid-uid",
            "username": "valid-user",
            "skip_user_id_check": False,
        }


@pytest.mark.usefixtures("_enabled_auth")
def test_is_user_authorized_invalid_token():
    """Test the is_user_authorized function with a mocked invalid-token."""
    with (
        patch("ols.src.auth.k8s.K8sClientSingleton.get_authn_api") as mock_authn_api,
        patch("ols.src.auth.k8s.K8sClientSingleton.get_authz_api") as mock_authz_api,
    ):
        # Setup mock responses for invalid token
        mock_authn_api.return_value.create_token_review.side_effect = (
            mock_token_review_response
        )
        mock_authz_api.return_value.create_subject_access_review.side_effect = (
            mock_subject_access_review_response
        )

        response = pytest.client.post(
            "/authorized",
            headers=[(b"authorization", b"Bearer invalid-token")],
        )
        assert response.status_code == 403
