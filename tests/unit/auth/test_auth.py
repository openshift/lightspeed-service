"""Unit tests for auth/auth module."""

import pytest

from ols.app.models.config import AuthenticationConfig, OLSConfig
from ols.src.auth import k8s, noop, noop_with_token
from ols.src.auth.auth import get_auth_dependency, use_k8s_auth


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


def test_get_auth_dependency_noop_with_token_module():
    """Test the function get_auth_dependency when module is set to no-op."""
    ols_config = OLSConfig()
    ols_config.authentication_config = AuthenticationConfig()
    ols_config.authentication_config.module = "noop-with-token"
    assert isinstance(
        get_auth_dependency(ols_config, "/path"), noop_with_token.AuthDependency
    )


def test_get_auth_dependency_unknown_module():
    """Test the function get_auth_dependency when module is set to unknown value."""
    ols_config = OLSConfig()
    ols_config.authentication_config = AuthenticationConfig()
    ols_config.authentication_config.module = "foo"
    with pytest.raises(
        Exception, match="Invalid/unknown auth. module was configured: foo"
    ):
        get_auth_dependency(ols_config, "/path")
