"""Unit tests for auth_dependency module."""

import os
from unittest.mock import patch

import pytest
from fastapi import HTTPException, Request
from kubernetes.client import AuthenticationV1Api, AuthorizationV1Api

from ols.utils import config
from ols.utils.auth_dependency import AuthDependency, K8sClientSingleton
from tests.mock_classes.mock_k8s_api import (
    mock_subject_access_review_response,
    mock_token_review_response,
)


@pytest.fixture(scope="module")
@patch.dict(os.environ, {"OLS_CONFIG_FILE": "tests/config/auth_config.yaml"})
def _setup():
    """Setups and load config."""
    global auth_dependency
    config.init_config("tests/config/auth_config.yaml")

    auth_dependency = AuthDependency(virtual_path="/ols-access")


def test_singleton_pattern(_setup):
    """Test if K8sClientSingleton is really a singleton."""
    k1 = K8sClientSingleton()
    k2 = K8sClientSingleton()
    assert k1 is k2


@pytest.mark.asyncio()
@patch("ols.utils.auth_dependency.K8sClientSingleton.get_authn_api")
@patch("ols.utils.auth_dependency.K8sClientSingleton.get_authz_api")
async def test_auth_dependency_valid_token(mock_authz_api, mock_authn_api, _setup):
    """Tests the auth dependency with a mocked valid-token."""
    # Setup mock responses for valid token
    mock_authn_api.return_value.create_token_review.side_effect = (
        mock_token_review_response
    )
    mock_authz_api.return_value.create_subject_access_review.side_effect = (
        mock_subject_access_review_response
    )

    # Simulate a request with a valid token
    request = Request(
        scope={"type": "http", "headers": [(b"authorization", b"Bearer valid-token")]}
    )

    user_uid, username = await auth_dependency(request)

    # Check if the correct user info has been returned
    assert user_uid == "valid-uid"
    assert username == "valid-user"


@pytest.mark.asyncio()
@patch("ols.utils.auth_dependency.K8sClientSingleton.get_authn_api")
@patch("ols.utils.auth_dependency.K8sClientSingleton.get_authz_api")
async def test_auth_dependency_invalid_token(mock_authz_api, mock_authn_api, _setup):
    """Test the auth dependency with a mocked invalid-token."""
    # Setup mock responses for invalid token
    mock_authn_api.return_value.create_token_review.side_effect = (
        mock_token_review_response
    )
    mock_authz_api.return_value.create_subject_access_review.side_effect = (
        mock_subject_access_review_response
    )

    # Simulate a request with an invalid token
    request = Request(
        scope={"type": "http", "headers": [(b"authorization", b"Bearer invalid-token")]}
    )

    # Expect an HTTPException for invalid tokens
    with pytest.raises(HTTPException) as exc_info:
        await auth_dependency(request)

    # Check if the correct status code is returned for unauthorized access
    assert exc_info.value.status_code == 403


@patch.dict(os.environ, {"KUBECONFIG": "tests/config/kubeconfig"})
def test_auth_dependency_config(_setup):
    """Test the auth dependency can load kubeconfig file."""
    from ols.utils.auth_dependency import K8sClientSingleton

    authn_client = K8sClientSingleton.get_authn_api()
    authz_client = K8sClientSingleton.get_authz_api()
    assert isinstance(
        authn_client, AuthenticationV1Api
    ), "authn_client is not an instance of AuthenticationV1Api"
    assert isinstance(
        authz_client, AuthorizationV1Api
    ), "authz_client is not an instance of AuthorizationV1Api"
