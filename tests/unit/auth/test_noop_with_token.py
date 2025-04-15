"""Unit tests for auth/noop module."""

from unittest.mock import patch

import pytest
from fastapi import HTTPException, Request

from ols.constants import DEFAULT_USER_NAME, DEFAULT_USER_UID, NO_USER_TOKEN
from ols.src.auth import noop_with_token


def test_noop_with_token_auth_dependency_path():
    """Check that the virtual path is stored in auth.dependency object."""
    path = "/ols-access"
    auth_dependency = noop_with_token.AuthDependency(virtual_path=path)
    assert auth_dependency.virtual_path == path


@pytest.mark.asyncio
async def test_noop_with_token_auth_dependency_token():
    """Check that the noop_with_token auth. dependency token."""
    path = "/ols-access"
    auth_dependency = noop_with_token.AuthDependency(virtual_path=path)

    # no auth header
    request = Request(scope={"type": "http", "headers": [], "query_string": ""})
    with pytest.raises(HTTPException, match="400: Bad request: No auth header found"):
        await auth_dependency(request)

    # no token
    request = Request(
        scope={
            "type": "http",
            "headers": [(b"authorization", b"bearer")],
            "query_string": "",
        }
    )
    with pytest.raises(
        HTTPException, match="400: Bad request: No token found in auth header"
    ):
        await auth_dependency(request)

    # token provided
    request = Request(
        scope={
            "type": "http",
            "headers": [(b"authorization", b"Bearer user-token")],
            "query_string": "",
        }
    )
    _, _, _, token = await auth_dependency(request)

    assert token == "user-token"  # noqa: S105


@pytest.mark.asyncio
async def test_noop_with_token_auth_dependency_call():
    """Check that the noop_with_token auth. dependency returns default user ID and name."""
    path = "/ols-access"
    auth_dependency = noop_with_token.AuthDependency(virtual_path=path)
    request = Request(
        scope={
            "type": "http",
            "headers": [(b"authorization", b"Bearer user-token")],
            "query_string": "",
        }
    )
    user_uid, username, skip_user_id_check, _ = await auth_dependency(request)

    assert user_uid == DEFAULT_USER_UID
    assert username == DEFAULT_USER_NAME
    assert skip_user_id_check is True


@pytest.mark.asyncio
async def test_noop_with_token_auth_dependency_call_disable_auth():
    """Check that the noop_with_token auth. dependency returns default user ID and name."""
    path = "/ols-access"
    with patch("ols.config.dev_config.disable_auth", True):
        auth_dependency = noop_with_token.AuthDependency(virtual_path=path)
        request = Request(
            scope={
                "type": "http",
                "headers": [],
                "query_string": "",
            }
        )
        user_uid, username, skip_user_id_check, token = await auth_dependency(request)

        assert user_uid == DEFAULT_USER_UID
        assert username == DEFAULT_USER_NAME
        assert skip_user_id_check is True
        assert token == NO_USER_TOKEN


@pytest.mark.asyncio
async def test_noop_with_token_auth_dependency_call_with_user_id():
    """Check that the noop_with_token auth. dependency returns provided user ID."""
    path = "/ols-access"
    auth_dependency = noop_with_token.AuthDependency(virtual_path=path)
    user_id_in_request = "00000000-1234-1234-1234-000000000000"
    request = Request(
        scope={
            "type": "http",
            "headers": [(b"authorization", b"Bearer user-token")],
            "query_string": f"user_id={user_id_in_request}",
        }
    )
    user_uid, username, skip_user_id_check, _ = await auth_dependency(request)

    assert user_uid == user_id_in_request
    assert username == DEFAULT_USER_NAME
    assert skip_user_id_check is True
