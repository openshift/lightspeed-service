"""Authentication related utilities."""

from ols.app.models.config import OLSConfig


def use_k8s_auth(ols_config: OLSConfig) -> bool:
    """Return True if k8s authentication should be used in the service."""
    if ols_config is None or ols_config.authentication_config is None:
        return False

    auth_module = ols_config.authentication_config.module
    return auth_module is not None and auth_module == "k8s"
