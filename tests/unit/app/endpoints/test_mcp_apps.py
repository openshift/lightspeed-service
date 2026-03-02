"""Unit tests for MCP Apps proxy endpoints."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from ols import config
from ols.constants import MCP_CLIENT_PLACEHOLDER, MCP_KUBERNETES_PLACEHOLDER

config.ols_config.authentication_config.module = "k8s"

from ols.app.endpoints.mcp_apps import (  # noqa: E402
    _get_server_config,
    call_mcp_app_tool,
    get_mcp_app_resource,
)
from ols.app.models.config import MCPServerConfig, MCPServers  # noqa: E402
from ols.app.models.models import (  # noqa: E402
    MCPAppResourceRequest,
    MCPAppToolCallRequest,
)

AUTH_TUPLE = ("test-uid", "test-user", False, "test-k8s-token")


@pytest.fixture()
def _mcp_servers():
    """Fixture that configures a test MCP server with static headers."""
    server = MCPServerConfig(
        name="test-server",
        url="http://test-server:8080/mcp",
    )
    server._resolved_headers = {"Authorization": "Bearer static-token"}
    original_servers = config.config.mcp_servers
    config.config.mcp_servers = MCPServers(servers=[server])
    if "mcp_servers_dict" in config.__dict__:
        del config.__dict__["mcp_servers_dict"]
    yield
    config.config.mcp_servers = original_servers
    if "mcp_servers_dict" in config.__dict__:
        del config.__dict__["mcp_servers_dict"]


@pytest.fixture()
def _mcp_servers_k8s_auth():
    """Fixture that configures an MCP server requiring kubernetes token auth."""
    server = MCPServerConfig(
        name="k8s-server",
        url="http://k8s-server:8080/mcp",
    )
    server._resolved_headers = {"Authorization": MCP_KUBERNETES_PLACEHOLDER}
    original_servers = config.config.mcp_servers
    config.config.mcp_servers = MCPServers(servers=[server])
    if "mcp_servers_dict" in config.__dict__:
        del config.__dict__["mcp_servers_dict"]
    yield
    config.config.mcp_servers = original_servers
    if "mcp_servers_dict" in config.__dict__:
        del config.__dict__["mcp_servers_dict"]


@pytest.fixture()
def _mcp_servers_client_auth():
    """Fixture that configures an MCP server requiring client-provided headers."""
    server = MCPServerConfig(
        name="client-server",
        url="http://client-server:8080/mcp",
    )
    server._resolved_headers = {"Authorization": MCP_CLIENT_PLACEHOLDER}
    original_servers = config.config.mcp_servers
    config.config.mcp_servers = MCPServers(servers=[server])
    if "mcp_servers_dict" in config.__dict__:
        del config.__dict__["mcp_servers_dict"]
    yield
    config.config.mcp_servers = original_servers
    if "mcp_servers_dict" in config.__dict__:
        del config.__dict__["mcp_servers_dict"]


def _mock_mcp_session_context(mock_session):
    """Build nested context managers for streamable_http_client + ClientSession.

    Returns a (transport_cm, session_factory) pair where session_factory
    accepts any args (like the real ClientSession constructor) and returns
    the mock session via an async context manager.
    """

    class _SessionCM:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, *args):
            pass

    class _TransportCM:
        async def __aenter__(self):
            return (AsyncMock(), AsyncMock(), None)

        async def __aexit__(self, *args):
            pass

    return _TransportCM(), _SessionCM


# ---------------------------------------------------------------------------
# _get_server_config
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_mcp_servers")
def test_get_server_config_found():
    """Test _get_server_config returns config for a known server."""
    cfg = _get_server_config("test-server")
    assert cfg["url"] == "http://test-server:8080/mcp"
    assert cfg["timeout"] == 30


@pytest.mark.usefixtures("_mcp_servers")
def test_get_server_config_not_found():
    """Test _get_server_config raises 404 for unknown server."""
    with pytest.raises(HTTPException) as exc_info:
        _get_server_config("nonexistent")
    assert exc_info.value.status_code == 404


def test_get_server_config_no_servers():
    """Test _get_server_config raises 404 when no MCP servers configured."""
    original = config.config.mcp_servers
    config.config.mcp_servers = MCPServers(servers=[])
    try:
        with pytest.raises(HTTPException) as exc_info:
            _get_server_config("any")
        assert exc_info.value.status_code == 404
    finally:
        config.config.mcp_servers = original


@pytest.mark.usefixtures("_mcp_servers_k8s_auth")
def test_get_server_config_resolves_kubernetes_token():
    """Test _get_server_config substitutes kubernetes placeholder with user token."""
    cfg = _get_server_config(
        "k8s-server", user_token="my-k8s-token"  # noqa: S106 # nosec
    )
    assert cfg["headers"]["Authorization"] == "Bearer my-k8s-token"


@pytest.mark.usefixtures("_mcp_servers_k8s_auth")
def test_get_server_config_missing_kubernetes_token():
    """Test _get_server_config raises 401 when kubernetes token is missing."""
    with pytest.raises(HTTPException) as exc_info:
        _get_server_config("k8s-server", user_token=None)
    assert exc_info.value.status_code == 401


@pytest.mark.usefixtures("_mcp_servers_client_auth")
def test_get_server_config_resolves_client_headers():
    """Test _get_server_config substitutes client placeholder with provided headers."""
    client_headers = {"client-server": {"Authorization": "Bearer client-tok"}}
    cfg = _get_server_config("client-server", client_headers=client_headers)
    assert cfg["headers"]["Authorization"] == "Bearer client-tok"


@pytest.mark.usefixtures("_mcp_servers_client_auth")
def test_get_server_config_missing_client_headers():
    """Test _get_server_config raises 401 when required client headers are missing."""
    with pytest.raises(HTTPException) as exc_info:
        _get_server_config("client-server", client_headers=None)
    assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# get_mcp_app_resource
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.usefixtures("_mcp_servers")
async def test_get_resource_invalid_uri():
    """Test resource endpoint rejects non-ui:// URIs."""
    request = MCPAppResourceRequest(
        resource_uri="https://evil.com", server_name="test-server"
    )
    with pytest.raises(HTTPException) as exc_info:
        await get_mcp_app_resource(request, auth=AUTH_TUPLE)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.usefixtures("_mcp_servers")
async def test_get_resource_empty_uri():
    """Test resource endpoint rejects bare 'ui://' with no path."""
    request = MCPAppResourceRequest(resource_uri="ui://", server_name="test-server")
    with pytest.raises(HTTPException) as exc_info:
        await get_mcp_app_resource(request, auth=AUTH_TUPLE)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.usefixtures("_mcp_servers")
async def test_get_resource_success():
    """Test successful resource fetch returns HTML content."""
    mock_content = SimpleNamespace(
        uri="ui://test-server/app.html",
        mimeType="text/html;profile=mcp-app",
        text="<html><body>Hello</body></html>",
        meta={"ui": {"csp": {"connectDomains": ["example.com"]}}},
    )
    mock_result = SimpleNamespace(contents=[mock_content])

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.read_resource = AsyncMock(return_value=mock_result)

    transport_cm, session_cls = _mock_mcp_session_context(mock_session)

    with (
        patch(
            "ols.app.endpoints.mcp_apps.streamable_http_client",
            return_value=transport_cm,
        ),
        patch("ols.app.endpoints.mcp_apps.ClientSession", session_cls),
    ):
        request = MCPAppResourceRequest(
            resource_uri="ui://test-server/app.html",
            server_name="test-server",
        )
        response = await get_mcp_app_resource(request, auth=AUTH_TUPLE)

    assert response.uri == "ui://test-server/app.html"
    assert response.mime_type == "text/html;profile=mcp-app"
    assert "<html>" in response.content
    assert response.content_type == "text"
    assert response.meta == {"ui": {"csp": {"connectDomains": ["example.com"]}}}


@pytest.mark.asyncio
@pytest.mark.usefixtures("_mcp_servers")
async def test_get_resource_empty_contents():
    """Test resource endpoint raises 404 when server returns empty contents."""
    mock_result = SimpleNamespace(contents=[])

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.read_resource = AsyncMock(return_value=mock_result)

    transport_cm, session_cls = _mock_mcp_session_context(mock_session)

    with (
        patch(
            "ols.app.endpoints.mcp_apps.streamable_http_client",
            return_value=transport_cm,
        ),
        patch("ols.app.endpoints.mcp_apps.ClientSession", session_cls),
    ):
        request = MCPAppResourceRequest(
            resource_uri="ui://test-server/missing.html",
            server_name="test-server",
        )
        with pytest.raises(HTTPException) as exc_info:
            await get_mcp_app_resource(request, auth=AUTH_TUPLE)
        assert exc_info.value.status_code == 404


@pytest.mark.asyncio
@pytest.mark.usefixtures("_mcp_servers")
async def test_get_resource_server_error():
    """Test resource endpoint returns 500 on MCP connection failure."""
    with patch(
        "ols.app.endpoints.mcp_apps.streamable_http_client",
        side_effect=ConnectionError("refused"),
    ):
        request = MCPAppResourceRequest(
            resource_uri="ui://test-server/app.html",
            server_name="test-server",
        )
        with pytest.raises(HTTPException) as exc_info:
            await get_mcp_app_resource(request, auth=AUTH_TUPLE)
        assert exc_info.value.status_code == 500


@pytest.mark.asyncio
@pytest.mark.usefixtures("_mcp_servers_k8s_auth")
async def test_get_resource_forwards_kubernetes_token():
    """Test resource endpoint forwards the user's kubernetes token to the MCP server."""
    mock_content = SimpleNamespace(
        uri="ui://k8s-server/app.html",
        mimeType="text/html",
        text="<html>ok</html>",
    )
    mock_result = SimpleNamespace(contents=[mock_content])

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.read_resource = AsyncMock(return_value=mock_result)

    transport_cm, session_cls = _mock_mcp_session_context(mock_session)

    with (
        patch(
            "ols.app.endpoints.mcp_apps.streamable_http_client",
            return_value=transport_cm,
        ),
        patch("ols.app.endpoints.mcp_apps.ClientSession", session_cls),
        patch("ols.app.endpoints.mcp_apps.httpx.AsyncClient") as mock_httpx,
    ):
        request = MCPAppResourceRequest(
            resource_uri="ui://k8s-server/app.html",
            server_name="k8s-server",
        )
        await get_mcp_app_resource(request, auth=AUTH_TUPLE)

    call_kwargs = mock_httpx.call_args
    assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer test-k8s-token"


# ---------------------------------------------------------------------------
# call_mcp_app_tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.usefixtures("_mcp_servers")
async def test_call_tool_success_text():
    """Test successful tool call returning text content."""
    text_block = SimpleNamespace(type="text", text="Pod count: 5")
    mock_result = SimpleNamespace(
        content=[text_block],
        structuredContent={"pods": 5},
        isError=False,
    )

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=mock_result)

    transport_cm, session_cls = _mock_mcp_session_context(mock_session)

    with (
        patch(
            "ols.app.endpoints.mcp_apps.streamable_http_client",
            return_value=transport_cm,
        ),
        patch("ols.app.endpoints.mcp_apps.ClientSession", session_cls),
    ):
        request = MCPAppToolCallRequest(
            server_name="test-server",
            tool_name="get-pods",
            arguments={"namespace": "default"},
        )
        response = await call_mcp_app_tool(request, auth=AUTH_TUPLE)

    assert len(response.content) == 1
    assert response.content[0] == {"type": "text", "text": "Pod count: 5"}
    assert response.structured_content == {"pods": 5}
    assert response.is_error is False


@pytest.mark.asyncio
@pytest.mark.usefixtures("_mcp_servers")
async def test_call_tool_success_image():
    """Test tool call returning image content block."""
    image_block = SimpleNamespace(type="image", data="base64data", mimeType="image/png")
    mock_result = SimpleNamespace(
        content=[image_block],
        structuredContent=None,
        isError=False,
    )

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=mock_result)

    transport_cm, session_cls = _mock_mcp_session_context(mock_session)

    with (
        patch(
            "ols.app.endpoints.mcp_apps.streamable_http_client",
            return_value=transport_cm,
        ),
        patch("ols.app.endpoints.mcp_apps.ClientSession", session_cls),
    ):
        request = MCPAppToolCallRequest(
            server_name="test-server",
            tool_name="render-chart",
        )
        response = await call_mcp_app_tool(request, auth=AUTH_TUPLE)

    assert response.content[0]["type"] == "image"
    assert response.content[0]["mimeType"] == "image/png"


@pytest.mark.asyncio
@pytest.mark.usefixtures("_mcp_servers")
async def test_call_tool_error_result():
    """Test tool call that returns isError=True."""
    text_block = SimpleNamespace(type="text", text="namespace not found")
    mock_result = SimpleNamespace(
        content=[text_block],
        structuredContent=None,
        isError=True,
    )

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=mock_result)

    transport_cm, session_cls = _mock_mcp_session_context(mock_session)

    with (
        patch(
            "ols.app.endpoints.mcp_apps.streamable_http_client",
            return_value=transport_cm,
        ),
        patch("ols.app.endpoints.mcp_apps.ClientSession", session_cls),
    ):
        request = MCPAppToolCallRequest(
            server_name="test-server",
            tool_name="get-pods",
            arguments={"namespace": "nope"},
        )
        response = await call_mcp_app_tool(request, auth=AUTH_TUPLE)

    assert response.is_error is True


@pytest.mark.asyncio
@pytest.mark.usefixtures("_mcp_servers")
async def test_call_tool_server_not_found():
    """Test tool call with unknown server name raises 404."""
    request = MCPAppToolCallRequest(
        server_name="nonexistent-server",
        tool_name="any-tool",
    )
    with pytest.raises(HTTPException) as exc_info:
        await call_mcp_app_tool(request, auth=AUTH_TUPLE)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
@pytest.mark.usefixtures("_mcp_servers")
async def test_call_tool_connection_error():
    """Test tool call returns 500 on MCP connection failure."""
    with patch(
        "ols.app.endpoints.mcp_apps.streamable_http_client",
        side_effect=ConnectionError("refused"),
    ):
        request = MCPAppToolCallRequest(
            server_name="test-server",
            tool_name="get-pods",
        )
        with pytest.raises(HTTPException) as exc_info:
            await call_mcp_app_tool(request, auth=AUTH_TUPLE)
        assert exc_info.value.status_code == 500


@pytest.mark.asyncio
@pytest.mark.usefixtures("_mcp_servers_k8s_auth")
async def test_call_tool_forwards_kubernetes_token():
    """Test tool call endpoint forwards the user's kubernetes token to the MCP server."""
    text_block = SimpleNamespace(type="text", text="ok")
    mock_result = SimpleNamespace(
        content=[text_block], structuredContent=None, isError=False
    )

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=mock_result)

    transport_cm, session_cls = _mock_mcp_session_context(mock_session)

    with (
        patch(
            "ols.app.endpoints.mcp_apps.streamable_http_client",
            return_value=transport_cm,
        ),
        patch("ols.app.endpoints.mcp_apps.ClientSession", session_cls),
        patch("ols.app.endpoints.mcp_apps.httpx.AsyncClient") as mock_httpx,
    ):
        request = MCPAppToolCallRequest(
            server_name="k8s-server",
            tool_name="get-pods",
        )
        await call_mcp_app_tool(request, auth=AUTH_TUPLE)

    call_kwargs = mock_httpx.call_args
    assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer test-k8s-token"


@pytest.mark.asyncio
@pytest.mark.usefixtures("_mcp_servers_client_auth")
async def test_call_tool_forwards_client_headers():
    """Test tool call endpoint forwards client-provided mcp_headers to the MCP server."""
    text_block = SimpleNamespace(type="text", text="ok")
    mock_result = SimpleNamespace(
        content=[text_block], structuredContent=None, isError=False
    )

    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=mock_result)

    transport_cm, session_cls = _mock_mcp_session_context(mock_session)

    with (
        patch(
            "ols.app.endpoints.mcp_apps.streamable_http_client",
            return_value=transport_cm,
        ),
        patch("ols.app.endpoints.mcp_apps.ClientSession", session_cls),
        patch("ols.app.endpoints.mcp_apps.httpx.AsyncClient") as mock_httpx,
    ):
        request = MCPAppToolCallRequest(
            server_name="client-server",
            tool_name="get-data",
            mcp_headers={"client-server": {"Authorization": "Bearer my-oauth-tok"}},
        )
        await call_mcp_app_tool(request, auth=AUTH_TUPLE)

    call_kwargs = mock_httpx.call_args
    assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer my-oauth-tok"
