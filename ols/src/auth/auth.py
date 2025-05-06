"""Authentication related utilities."""

import logging

from ols.app.models.config import OLSConfig

from . import k8s, noop, noop_with_token
from .auth_dependency_interface import AuthDependencyInterface

logger = logging.getLogger(__name__)


def use_k8s_auth(ols_config: OLSConfig) -> bool:
    """Return True if k8s authentication should be used in the service."""
    auth_module = ols_config.authentication_config.module
    return auth_module is not None and auth_module == "k8s"


def get_auth_dependency(
    ols_config: OLSConfig, virtual_path: str
) -> AuthDependencyInterface:
    """Select the configured authentication dependency interface."""
    module = ols_config.authentication_config.module
    if module is None:
        raise Exception("Authentication module is not specified")

    # module is specified -> time to construct AuthDependency instance
    logger.info(
        "Authentication retrieval for module %s and virtual path %s",
        module,
        virtual_path,
    )

    match module:
        case "k8s":
            return k8s.AuthDependency(virtual_path=virtual_path)
        case "noop":
            return noop.AuthDependency(virtual_path=virtual_path)
        case "noop-with-token":
            return noop_with_token.AuthDependency(virtual_path=virtual_path)
        case _:
            # this is internal error and should not happen in reality
            raise Exception(f"Invalid/unknown auth. module was configured: {module}")
