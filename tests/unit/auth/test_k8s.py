"""Unit tests for auth/k8s module."""

import os
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, Request
from kubernetes.client import AuthenticationV1Api, AuthorizationV1Api
from kubernetes.client.rest import ApiException

from ols import config
from ols.src.auth.k8s import (
    CLUSTER_ID_LOCAL,
    AuthDependency,
    ClusterIDUnavailableError,
    K8sClientSingleton,
)
from tests.mock_classes.mock_k8s_api import (
    MockK8sResponseStatus,
    mock_subject_access_review_response,
    mock_token_review_response,
)

auth_dependency: Optional[AuthDependency] = None


@pytest.fixture(scope="function")
def _setup():
    """Setups and load config."""
    global auth_dependency
    config.reload_from_yaml_file("tests/config/auth_config.yaml")
    auth_dependency = AuthDependency(virtual_path="/ols-access")


@pytest.mark.usefixtures("_setup")
def test_singleton_pattern():
    """Test if K8sClientSingleton is really a singleton."""
    k1 = K8sClientSingleton()
    k2 = K8sClientSingleton()
    assert k1 is k2


@pytest.mark.usefixtures("_setup")
@pytest.mark.asyncio
@patch("ols.src.auth.k8s.K8sClientSingleton.get_authn_api")
@patch("ols.src.auth.k8s.K8sClientSingleton.get_authz_api")
async def test_auth_dependency_valid_token(mock_authz_api, mock_authn_api):
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


@pytest.mark.usefixtures("_setup")
@pytest.mark.asyncio
@patch("ols.src.auth.k8s.K8sClientSingleton.get_authn_api")
@patch("ols.src.auth.k8s.K8sClientSingleton.get_authz_api")
async def test_auth_dependency_invalid_token(mock_authz_api, mock_authn_api):
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


@pytest.mark.usefixtures("_setup")
@pytest.mark.asyncio
@patch("ols.src.auth.k8s.K8sClientSingleton.get_authz_api")
async def test_cluster_id_is_used_for_kube_admin(mock_authz_api):
    """Test the cluster id is used as user_id when user is kube:admin."""
    mock_authz_api.return_value.create_subject_access_review.side_effect = (
        mock_subject_access_review_response
    )

    # simulate a request with a valid token
    request = Request(
        scope={"type": "http", "headers": [(b"authorization", b"Bearer valid-token")]}
    )

    with (
        patch(
            "ols.src.auth.k8s.get_user_info",
            return_value=MockK8sResponseStatus(
                True, True, "kube:admin", "some-uuid", "ols-group"
            ),
        ),
        patch(
            "ols.src.auth.k8s.K8sClientSingleton.get_cluster_id",
            return_value="some-cluster-id",
        ),
    ):
        user_uid, username = await auth_dependency(request)

    # check if the correct user info has been returned
    assert user_uid == "some-cluster-id"
    assert username == "kube:admin"


@pytest.mark.usefixtures("_setup")
@patch.dict(os.environ, {"KUBECONFIG": "tests/config/kubeconfig"})
def test_auth_dependency_config():
    """Test the auth dependency can load kubeconfig file."""
    from ols.src.auth.k8s import K8sClientSingleton

    authn_client = K8sClientSingleton.get_authn_api()
    authz_client = K8sClientSingleton.get_authz_api()
    assert isinstance(
        authn_client, AuthenticationV1Api
    ), "authn_client is not an instance of AuthenticationV1Api"
    assert isinstance(
        authz_client, AuthorizationV1Api
    ), "authz_client is not an instance of AuthorizationV1Api"


@patch("ols.src.auth.k8s.K8sClientSingleton.get_custom_objects_api")
def test_get_cluster_id(mock_get_custom_objects_api):
    """Test get_cluster_id function."""
    cluster_id = {"spec": {"clusterID": "some-cluster-id"}}
    mocked_call = MagicMock()
    mocked_call.get_cluster_custom_object.return_value = cluster_id
    mock_get_custom_objects_api.return_value = mocked_call
    assert K8sClientSingleton._get_cluster_id() == "some-cluster-id"

    # keyerror
    cluster_id = {"spec": {}}
    mocked_call = MagicMock()
    mocked_call.get_cluster_custom_object.return_value = cluster_id
    mock_get_custom_objects_api.return_value = mocked_call
    with pytest.raises(ClusterIDUnavailableError, match="Failed to get cluster ID"):
        K8sClientSingleton._get_cluster_id()

    # typeerror
    cluster_id = None
    mocked_call = MagicMock()
    mocked_call.get_cluster_custom_object.return_value = cluster_id
    mock_get_custom_objects_api.return_value = mocked_call
    with pytest.raises(ClusterIDUnavailableError, match="Failed to get cluster ID"):
        K8sClientSingleton._get_cluster_id()

    # typeerror
    mock_get_custom_objects_api.side_effect = ApiException()
    with pytest.raises(ClusterIDUnavailableError, match="Failed to get cluster ID"):
        K8sClientSingleton._get_cluster_id()

    # exception
    mock_get_custom_objects_api.side_effect = Exception()
    with pytest.raises(ClusterIDUnavailableError, match="Failed to get cluster ID"):
        K8sClientSingleton._get_cluster_id()


@patch("ols.src.auth.k8s.RUNNING_IN_CLUSTER", True)
@patch("ols.src.auth.k8s.K8sClientSingleton.__new__")
@patch("ols.src.auth.k8s.K8sClientSingleton._get_cluster_id")
def test_get_cluster_id_in_cluster(mock_get_cluster_id, _mock_new):
    """Test get_cluster_id function when running inside of cluster."""
    mock_get_cluster_id.return_value = "some-cluster-id"
    assert K8sClientSingleton.get_cluster_id() == "some-cluster-id"


@patch("ols.src.auth.k8s.RUNNING_IN_CLUSTER", False)
@patch("ols.src.auth.k8s.K8sClientSingleton.__new__")
def test_get_cluster_id_outside_of_cluster(_mock_new):
    """Test get_cluster_id function when running outside of cluster."""
    # ensure cluster_id is None to trigger the condition
    K8sClientSingleton._cluster_id = None
    assert K8sClientSingleton.get_cluster_id() == CLUSTER_ID_LOCAL
