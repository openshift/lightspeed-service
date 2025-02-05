"""Manage authentication flow for FastAPI endpoints with no-op auth."""

import logging

from fastapi import Request

from ols import config
from ols.constants import DEFAULT_USER_NAME, DEFAULT_USER_UID

from .auth_dependency_interface import AuthDependencyInterface

logger = logging.getLogger(__name__)


class AuthDependency(AuthDependencyInterface):
    """Create an AuthDependency Class that allows customizing the acces Scope path to check."""

    def __init__(self, virtual_path: str = "/ols-access") -> None:
        """Initialize the required allowed paths for authorization checks."""
        self.virtual_path = virtual_path
        # skip user_id suid check if noop auth to allow consumers provide user_id
        self.skip_userid_check = True

    async def __call__(self, request: Request) -> tuple[str, str, bool]:
        """Validate FastAPI Requests for authentication and authorization.

        Args:
            request: The FastAPI request object.

        Returns:
            The user's UID and username if authentication and authorization succeed
            user_id check is skipped with noop auth to allow consumers provide user_id
        """
        if config.dev_config.disable_auth:
            if (
                config.ols_config.logging_config is None
                or not config.ols_config.logging_config.suppress_auth_checks_warning_in_log
            ):
                logger.warning("Auth checks disabled, skipping")
            # Use constant user ID and user name in case auth. is disabled
            # It will be needed for testing purposes because (for example)
            # conversation history and user feedback depend on having any
            # user ID (identity) in proper format (UUID)
            return DEFAULT_USER_UID, DEFAULT_USER_NAME, self.skip_userid_check

        logger.warning("Using no-op dependency authentication!")
        logger.warning(
            "The service is in insecure mode meant only to be used in devel environment"
        )
        # try to read user ID from request
        user_id = request.query_params.get("user_id", DEFAULT_USER_UID)
        logger.info("User ID: %s", user_id)
        return user_id, DEFAULT_USER_NAME, self.skip_userid_check
