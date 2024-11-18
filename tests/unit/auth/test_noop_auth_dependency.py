"""Unit tests for auth/noop module."""

import pytest
from fastapi import Request

from ols.constants import DEFAULT_USER_NAME, DEFAULT_USER_UID
from ols.src.auth import noop


def test_noop_auth_dependency_path():
    """Check that the virtual path is stored in auth.dependency object."""
    path = "/ols-access"
    auth_dependency = noop.AuthDependency(virtual_path=path)
    assert auth_dependency.virtual_path == path


@pytest.mark.asyncio
async def test_noop_auth_dependency_call():
    """Check that the no-op auth. dependency returns default user ID and name as expected."""
    path = "/ols-access"
    auth_dependency = noop.AuthDependency(virtual_path=path)
    # Simulate a request without a token
    request = Request(scope={"type": "http", "headers": []})
    user_uid, username = await auth_dependency(request)

    # Check if the correct user info has been returned
    assert user_uid == DEFAULT_USER_UID
    assert username == DEFAULT_USER_NAME
