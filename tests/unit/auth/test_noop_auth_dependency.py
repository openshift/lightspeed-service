"""Unit tests for auth/noop module."""

from unittest.mock import patch

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
    # Simulate a request without a token nor user_id parameter
    request = Request(scope={"type": "http", "headers": [], "query_string": ""})
    user_uid, username, skip_user_id_check = await auth_dependency(request)

    # Check if the correct user info has been returned
    assert user_uid == DEFAULT_USER_UID
    assert username == DEFAULT_USER_NAME
    assert skip_user_id_check is True


@pytest.mark.asyncio
@patch("ols.config.dev_config.disable_auth", True)
async def test_noop_auth_dependency_call_disable_auth():
    """Check that the no-op auth. dependency returns default user ID and name as expected."""
    path = "/ols-access"
    auth_dependency = noop.AuthDependency(virtual_path=path)
    # Simulate a request without a token nor user_id parameter
    request = Request(scope={"type": "http", "headers": [], "query_string": ""})
    user_uid, username, skip_user_id_check = await auth_dependency(request)

    # Check if the correct user info has been returned
    assert user_uid == DEFAULT_USER_UID
    assert username == DEFAULT_USER_NAME
    assert skip_user_id_check is True


@pytest.mark.asyncio
async def test_noop_auth_dependency_call_with_user_id():
    """Check that the no-op auth. dependency returns provided user ID."""
    path = "/ols-access"
    auth_dependency = noop.AuthDependency(virtual_path=path)
    # Simulate a request with user_id specified as optional parameter
    user_id_in_request = "00000000-1234-1234-1234-000000000000"
    request = Request(
        scope={
            "type": "http",
            "headers": [],
            "query_string": f"user_id={user_id_in_request}",
        }
    )
    user_uid, username, skip_user_id_check = await auth_dependency(request)

    # Check if the correct user info has been returned
    assert user_uid == user_id_in_request
    assert username == DEFAULT_USER_NAME
    assert skip_user_id_check is True
