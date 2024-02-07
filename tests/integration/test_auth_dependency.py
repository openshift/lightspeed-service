import pytest
from fastapi import HTTPException, Request
from unittest.mock import patch, MagicMock
from ols.utils.auth_dependency import auth_dependency


@pytest.mark.asyncio
@patch("ols.utils.auth_dependency._k8s_auth", return_value=True)
async def test_auth_dependency_success(mock_k8s_auth):
    request = MagicMock(spec=Request)
    request.headers.get.return_value = "Bearer valid-token"
    await auth_dependency(request)  
    mock_k8s_auth.assert_called_once_with("valid-token")


@pytest.mark.asyncio
@patch("ols.utils.auth_dependency._k8s_auth", return_value=False)
async def test_auth_dependency_failure(mock_k8s_auth):
    """Test auth_dependency correctly handles an unauthorized request."""
    # Mock a Request with an invalid Authorization header
    request = MagicMock(spec=Request)
    request.headers.get.return_value = "Bearer invalid-token"

    # Expect an HTTPException for an invalid request
    with pytest.raises(HTTPException) as exc_info:
        await auth_dependency(request)

    assert exc_info.value.status_code == 403
    mock_k8s_auth.assert_called_once_with("invalid-token")

@pytest.mark.asyncio
async def test_auth_dependency_missing_header():
    """Test auth_dependency raises an exception when the Authorization header is missing."""
    # Mock a Request without an Authorization header
    request = MagicMock(spec=Request)
    request.headers.get.return_value = None

    # Expect an HTTPException due to missing Authorization header
    with pytest.raises(HTTPException) as exc_info:
        await auth_dependency(request)

    assert exc_info.value.status_code == 401
    assert "No auth header found" in exc_info.value.detail
