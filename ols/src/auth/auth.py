"""Authentication related utilities."""

from abc import ABC, abstractmethod

from fastapi import Request

from ols.app.models.config import OLSConfig


def use_k8s_auth(ols_config: OLSConfig) -> bool:
    """Return True if k8s authentication should be used in the service."""
    if ols_config is None or ols_config.authentication_config is None:
        return False

    auth_module = ols_config.authentication_config.module
    return auth_module is not None and auth_module == "k8s"


class AuthDependencyInterface(ABC):
    """An interface to be satisfied by all auth. implementations."""

    @abstractmethod
    async def __call__(self, request: Request) -> tuple[str, str]:
        """Validate FastAPI Requests for authentication and authorization."""
        return ("", "")
