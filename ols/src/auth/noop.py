"""Manage authentication flow for FastAPI endpoints with no-op auth."""

import logging

from fastapi import Request

from ols.constants import DEFAULT_USER_NAME, DEFAULT_USER_UID

from .auth import AuthDependencyInterface

logger = logging.getLogger(__name__)


class AuthDependency(AuthDependencyInterface):
    """Create an AuthDependency Class that allows customizing the acces Scope path to check."""

    def __init__(self, virtual_path: str = "/ols-access") -> None:
        """Initialize the required allowed paths for authorization checks."""
        self.virtual_path = virtual_path

    async def __call__(self, request: Request) -> tuple[str, str]:
        """Validate FastAPI Requests for authentication and authorization.

        Args:
            request: The FastAPI request object.

        Returns:
            The user's UID and username if authentication and authorization succeed.
        """
        return DEFAULT_USER_UID, DEFAULT_USER_NAME
