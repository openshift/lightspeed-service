"""Unit tests for auth/auth module."""

from ols.app.models.config import AuthenticationConfig, OLSConfig
from ols.src.auth.auth import use_k8s_auth


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
