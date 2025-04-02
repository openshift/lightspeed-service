"""Unit tests for auth/auth module."""

import pytest
from fastapi import HTTPException, Request

from ols.app.models.config import AuthenticationConfig, OLSConfig
from ols.src.auth import k8s, noop
from ols.src.auth.auth import get_auth_dependency, use_k8s_auth
from ols.src.auth.auth_dependency_interface import (
    extract_bearer_token,
    extract_token_from_request,
)


def test_use_k8s_auth_no_auth_config():
    """Test the function use_k8s_auth."""
    ols_config = OLSConfig()
    ols_config.authentication_config = None
    assert use_k8s_auth(ols_config) is False


def test_use_k8s_auth_default_auth_config():
    """Test the function use_k8s_auth."""
    ols_config = OLSConfig()
    ols_config.authentication_config = AuthenticationConfig()
    assert use_k8s_auth(ols_config) is False


def test_use_k8s_auth_k8s_module():
    """Test the function use_k8s_auth when k8s module is selected."""
    ols_config = OLSConfig()
    ols_config.authentication_config = AuthenticationConfig()
    ols_config.authentication_config.module = "k8s"
    assert use_k8s_auth(ols_config) is True


def test_use_k8s_auth_no_k8s_module():
    """Test the function use_k8s_auth when module different from k8s is selected."""
    ols_config = OLSConfig()
    ols_config.authentication_config = AuthenticationConfig()
    ols_config.authentication_config.module = "foo"
    assert use_k8s_auth(ols_config) is False


def test_get_auth_dependency_no_auth_config():
    """Test the function get_auth_dependency when auth config is not setup."""
    ols_config = OLSConfig()
    ols_config.authentication_config = None
    with pytest.raises(Exception, match="Authentication is not configured properly"):
        get_auth_dependency(ols_config, "/path")


def test_get_auth_dependency_default_auth_config():
    """Test the function get_auth_dependency when auth config module is not explicitly setup."""
    ols_config = OLSConfig()
    ols_config.authentication_config = AuthenticationConfig()
    with pytest.raises(Exception, match="Authentication module is not specified"):
        get_auth_dependency(ols_config, "/path")


def test_get_auth_dependency_noop_module():
    """Test the function get_auth_dependency when module is set to no-op."""
    ols_config = OLSConfig()
    ols_config.authentication_config = AuthenticationConfig()
    ols_config.authentication_config.module = "noop"
    assert isinstance(get_auth_dependency(ols_config, "/path"), noop.AuthDependency)


def test_get_auth_dependency_k8s_module():
    """Test the function get_auth_dependency when module is set to k8s."""
    ols_config = OLSConfig()
    ols_config.authentication_config = AuthenticationConfig()
    ols_config.authentication_config.module = "k8s"
    assert isinstance(get_auth_dependency(ols_config, "/path"), k8s.AuthDependency)


def test_get_auth_dependency_unknown_module():
    """Test the function get_auth_dependency when module is set to unknown value."""
    ols_config = OLSConfig()
    ols_config.authentication_config = AuthenticationConfig()
    ols_config.authentication_config.module = "foo"
    with pytest.raises(
        Exception, match="Invalid/unknown auth. module was configured: foo"
    ):
        get_auth_dependency(ols_config, "/path")


def test_extract_bearer_token():
    """Test the function extract_bearer_token."""
    # good value
    assert extract_bearer_token("Bearer token") == "token"

    # bad values
    assert extract_bearer_token("Bearer") == ""
    assert extract_bearer_token("sha256~aaaaaa") == ""


def test_extract_token_from_request():
    """Test the function extract_token_from_request."""
    request = Request(
        scope={"type": "http", "headers": [(b"authorization", b"Bearer token")]}
    )
    assert extract_token_from_request(request) == "token"

    with pytest.raises(HTTPException, match="Unauthorized: No auth header found"):
        extract_token_from_request(Request(scope={"type": "http", "headers": []}))
