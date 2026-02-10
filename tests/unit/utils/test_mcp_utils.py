"""Unit tests for MCP utilities."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools.structured import StructuredTool

from ols import constants
from ols.app.models.config import MCPServerConfig
from ols.utils.mcp_utils import (
    build_mcp_config,
    gather_mcp_tools,
    get_mcp_tools,
    get_servers_requiring_client_headers,
    resolve_header_value,
    resolve_server_headers,
)


@pytest.fixture
def mock_mcp_server():
    """Create a mock MCP server config."""
    server = MagicMock(spec=MCPServerConfig)
    server.name = "test-server"
    server.url = "http://test:8080/mcp"
    server.timeout = None
    server.headers = {}
    server.resolved_headers = {}
    return server


@pytest.fixture
def mock_k8s_server():
    """Create a mock MCP server with k8s authentication."""
    server = MagicMock(spec=MCPServerConfig)
    server.name = "k8s-server"
    server.url = "http://k8s:8080/mcp"
    server.timeout = 30
    server.headers = {"Authorization": constants.MCP_KUBERNETES_PLACEHOLDER}
    server.resolved_headers = {"Authorization": constants.MCP_KUBERNETES_PLACEHOLDER}
    return server


@pytest.fixture
def mock_client_server():
    """Create a mock MCP server with client authentication."""
    server = MagicMock(spec=MCPServerConfig)
    server.name = "client-server"
    server.url = "http://client:8080/mcp"
    server.timeout = None
    server.headers = {"Authorization": constants.MCP_CLIENT_PLACEHOLDER}
    server.resolved_headers = {"Authorization": constants.MCP_CLIENT_PLACEHOLDER}
    return server


@pytest.fixture
def mock_file_server():
    """Create a mock MCP server with file-based authentication."""
    server = MagicMock(spec=MCPServerConfig)
    server.name = "file-server"
    server.url = "http://file:8080/mcp"
    server.timeout = None
    server.headers = {"Authorization": "Bearer file-token"}
    server.resolved_headers = {"Authorization": "Bearer file-token"}
    return server


@pytest.fixture
def mock_tool():
    """Create a mock StructuredTool."""
    tool = MagicMock(spec=StructuredTool)
    tool.name = "test_tool"
    tool.description = "A test tool"
    tool.metadata = {}
    return tool


class TestGetServersRequiringClientHeaders:
    """Tests for get_servers_requiring_client_headers function."""

    def test_empty_servers(self):
        """Test with no servers configured."""
        result = get_servers_requiring_client_headers(None)
        assert result == {}

    def test_no_client_auth_servers(self, mock_k8s_server, mock_file_server):
        """Test with servers that don't require client headers."""
        mock_servers = MagicMock()
        mock_servers.servers = [mock_k8s_server, mock_file_server]

        result = get_servers_requiring_client_headers(mock_servers)
        assert result == {}

    def test_single_client_auth_server(self, mock_client_server):
        """Test with one server requiring client headers."""
        mock_servers = MagicMock()
        mock_servers.servers = [mock_client_server]

        result = get_servers_requiring_client_headers(mock_servers)
        assert result == {"client-server": ["Authorization"]}

    def test_mixed_servers(self, mock_k8s_server, mock_client_server, mock_file_server):
        """Test with mix of server types."""
        mock_servers = MagicMock()
        mock_servers.servers = [mock_k8s_server, mock_client_server, mock_file_server]

        result = get_servers_requiring_client_headers(mock_servers)
        assert result == {"client-server": ["Authorization"]}

    def test_multiple_client_headers(self):
        """Test server requiring multiple client headers."""
        server = MagicMock(spec=MCPServerConfig)
        server.name = "multi-header-server"
        server.headers = {
            "Authorization": constants.MCP_CLIENT_PLACEHOLDER,
            "X-Custom": constants.MCP_CLIENT_PLACEHOLDER,
        }
        server.resolved_headers = {
            "Authorization": constants.MCP_CLIENT_PLACEHOLDER,
            "X-Custom": constants.MCP_CLIENT_PLACEHOLDER,
        }

        mock_servers = MagicMock()
        mock_servers.servers = [server]

        result = get_servers_requiring_client_headers(mock_servers)
        assert result == {"multi-header-server": ["Authorization", "X-Custom"]}


class TestResolveHeaderValue:
    """Tests for resolve_header_value function."""

    def test_kubernetes_placeholder_with_token(self):
        """Test k8s placeholder resolution with token."""
        result = resolve_header_value(
            constants.MCP_KUBERNETES_PLACEHOLDER,
            "Authorization",
            "test-server",
            "k8s-token-123",
            None,
        )
        assert result == "Bearer k8s-token-123"

    def test_kubernetes_placeholder_without_token(self):
        """Test k8s placeholder resolution without token."""
        result = resolve_header_value(
            constants.MCP_KUBERNETES_PLACEHOLDER,
            "Authorization",
            "test-server",
            None,
            None,
        )
        assert result is None

    def test_client_placeholder_with_headers(self):
        """Test client placeholder resolution with headers."""
        client_headers = {"test-server": {"Authorization": "Bearer client-token"}}
        result = resolve_header_value(
            constants.MCP_CLIENT_PLACEHOLDER,
            "Authorization",
            "test-server",
            None,
            client_headers,
        )
        assert result == "Bearer client-token"

    def test_client_placeholder_without_headers(self):
        """Test client placeholder resolution without headers."""
        result = resolve_header_value(
            constants.MCP_CLIENT_PLACEHOLDER,
            "Authorization",
            "test-server",
            None,
            None,
        )
        assert result is None

    def test_client_placeholder_missing_server(self):
        """Test client placeholder with wrong server name."""
        client_headers = {"other-server": {"Authorization": "Bearer token"}}
        result = resolve_header_value(
            constants.MCP_CLIENT_PLACEHOLDER,
            "Authorization",
            "test-server",
            None,
            client_headers,
        )
        assert result is None

    def test_client_placeholder_missing_header(self):
        """Test client placeholder with missing header name."""
        client_headers = {"test-server": {"X-Custom": "value"}}
        result = resolve_header_value(
            constants.MCP_CLIENT_PLACEHOLDER,
            "Authorization",
            "test-server",
            None,
            client_headers,
        )
        assert result is None

    def test_already_resolved_value(self):
        """Test already resolved header value."""
        result = resolve_header_value(
            "Bearer file-token", "Authorization", "test-server", None, None
        )
        assert result == "Bearer file-token"


class TestResolveServerHeaders:
    """Tests for resolve_server_headers function."""

    def test_k8s_auth_with_token(self, mock_k8s_server):
        """Test k8s server with token."""
        result = resolve_server_headers(mock_k8s_server, "k8s-token", None)
        assert result == {"Authorization": "Bearer k8s-token"}

    def test_k8s_auth_without_token(self, mock_k8s_server):
        """Test k8s server without token."""
        result = resolve_server_headers(mock_k8s_server, None, None)
        assert result is None

    def test_client_auth_with_headers(self, mock_client_server):
        """Test client server with headers."""
        client_headers = {"client-server": {"Authorization": "Bearer client-token"}}
        result = resolve_server_headers(mock_client_server, None, client_headers)
        assert result == {"Authorization": "Bearer client-token"}

    def test_client_auth_without_headers(self, mock_client_server):
        """Test client server without headers."""
        result = resolve_server_headers(mock_client_server, None, None)
        assert result is None

    def test_file_auth(self, mock_file_server):
        """Test file-based auth server."""
        result = resolve_server_headers(mock_file_server, None, None)
        assert result == {"Authorization": "Bearer file-token"}

    def test_no_headers(self, mock_mcp_server):
        """Test server without headers."""
        result = resolve_server_headers(mock_mcp_server, None, None)
        assert result == {}


class TestBuildMcpConfig:
    """Tests for build_mcp_config function."""

    def test_empty_servers_list(self):
        """Test with empty servers list."""
        result = build_mcp_config([], None, None)
        assert result == {}

    def test_single_k8s_server(self, mock_k8s_server):
        """Test with single k8s server."""
        result = build_mcp_config([mock_k8s_server], "k8s-token", None)
        assert "k8s-server" in result
        assert result["k8s-server"]["transport"] == "streamable_http"
        assert result["k8s-server"]["url"] == "http://k8s:8080/mcp"
        assert result["k8s-server"]["headers"] == {"Authorization": "Bearer k8s-token"}
        assert result["k8s-server"]["timeout"] == 30

    def test_single_client_server(self, mock_client_server):
        """Test with single client server."""
        client_headers = {"client-server": {"Authorization": "Bearer client-token"}}
        result = build_mcp_config([mock_client_server], None, client_headers)
        assert "client-server" in result
        assert result["client-server"]["headers"] == {
            "Authorization": "Bearer client-token"
        }

    def test_single_file_server(self, mock_file_server):
        """Test with single file server."""
        result = build_mcp_config([mock_file_server], None, None)
        assert "file-server" in result
        assert result["file-server"]["headers"] == {
            "Authorization": "Bearer file-token"
        }

    def test_mixed_servers(self, mock_k8s_server, mock_client_server, mock_file_server):
        """Test with multiple server types."""
        client_headers = {"client-server": {"Authorization": "Bearer client-token"}}
        result = build_mcp_config(
            [mock_k8s_server, mock_client_server, mock_file_server],
            "k8s-token",
            client_headers,
        )
        assert len(result) == 3
        assert "k8s-server" in result
        assert "client-server" in result
        assert "file-server" in result

    def test_skips_unresolvable_server(self, mock_k8s_server, mock_file_server):
        """Test that unresolvable servers are skipped."""
        # k8s server without token should be skipped
        result = build_mcp_config([mock_k8s_server, mock_file_server], None, None)
        assert len(result) == 1
        assert "file-server" in result
        assert "k8s-server" not in result

    def test_server_without_timeout(self, mock_client_server):
        """Test server config without timeout."""
        client_headers = {"client-server": {"Authorization": "Bearer token"}}
        result = build_mcp_config([mock_client_server], None, client_headers)
        assert "timeout" not in result["client-server"]

    def test_error_handling(self):
        """Test error handling in config building."""
        # Create a server that will cause an error
        bad_server = MagicMock(spec=MCPServerConfig)
        bad_server.name = "bad-server"
        bad_server.resolved_headers.items.side_effect = Exception("Test error")

        result = build_mcp_config([bad_server], None, None)
        assert result == {}


@pytest.mark.asyncio
class TestGatherMcpTools:
    """Tests for gather_mcp_tools function."""

    async def test_empty_servers(self):
        """Test with no servers."""
        result = await gather_mcp_tools({})
        assert result == []

    async def test_single_server_success(self, mock_tool):
        """Test gathering tools from one server."""
        with patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get_tools.return_value = [mock_tool]
            mock_client_cls.return_value = mock_client

            servers = {
                "test-server": {"transport": "streamable_http", "url": "http://test"}
            }
            result = await gather_mcp_tools(servers)

            assert len(result) == 1
            assert result[0].name == "test_tool"
            assert result[0].metadata["mcp_server"] == "test-server"

    async def test_multiple_servers(self, mock_tool):
        """Test gathering tools from multiple servers."""
        with patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_client_cls:
            tool1 = MagicMock(spec=StructuredTool)
            tool1.name = "tool1"
            tool1.metadata = {}

            tool2 = MagicMock(spec=StructuredTool)
            tool2.name = "tool2"
            tool2.metadata = {}

            mock_client = AsyncMock()
            mock_client.get_tools.side_effect = [[tool1], [tool2]]
            mock_client_cls.return_value = mock_client

            servers = {
                "server1": {"transport": "streamable_http", "url": "http://s1"},
                "server2": {"transport": "streamable_http", "url": "http://s2"},
            }
            result = await gather_mcp_tools(servers)

            assert len(result) == 2
            assert result[0].metadata["mcp_server"] == "server1"
            assert result[1].metadata["mcp_server"] == "server2"

    async def test_server_failure_isolation(self):
        """Test that one server failure doesn't affect others."""
        with patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_client_cls:
            tool = MagicMock(spec=StructuredTool)
            tool.name = "good_tool"
            tool.metadata = {}

            mock_client = AsyncMock()
            mock_client.get_tools.side_effect = [
                Exception("Server 1 failed"),
                [tool],
            ]
            mock_client_cls.return_value = mock_client

            servers = {
                "bad-server": {"transport": "streamable_http", "url": "http://bad"},
                "good-server": {"transport": "streamable_http", "url": "http://good"},
            }
            result = await gather_mcp_tools(servers)

            assert len(result) == 1
            assert result[0].name == "good_tool"
            assert result[0].metadata["mcp_server"] == "good-server"

    async def test_tool_filtering_with_allowlist(self):
        """Test filtering tools by allowed names."""
        with patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_client_cls:
            tool1 = MagicMock(spec=StructuredTool)
            tool1.name = "allowed_tool"
            tool1.metadata = {}

            tool2 = MagicMock(spec=StructuredTool)
            tool2.name = "blocked_tool"
            tool2.metadata = {}

            mock_client = AsyncMock()
            mock_client.get_tools.return_value = [tool1, tool2]
            mock_client_cls.return_value = mock_client

            servers = {
                "test-server": {"transport": "streamable_http", "url": "http://test"}
            }
            result = await gather_mcp_tools(
                servers, allowed_tool_names={"allowed_tool"}
            )

            assert len(result) == 1
            assert result[0].name == "allowed_tool"

    async def test_tool_without_metadata(self):
        """Test handling tools without metadata attribute."""
        with patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_client_cls:
            # Tool without metadata attribute
            tool = MagicMock(spec=StructuredTool)
            tool.name = "test_tool"
            del tool.metadata

            mock_client = AsyncMock()
            mock_client.get_tools.return_value = [tool]
            mock_client_cls.return_value = mock_client

            servers = {
                "test-server": {"transport": "streamable_http", "url": "http://test"}
            }
            result = await gather_mcp_tools(servers)

            assert len(result) == 1
            assert hasattr(result[0], "metadata")
            assert result[0].metadata["mcp_server"] == "test-server"

    async def test_tool_with_none_metadata(self):
        """Test handling tools with None metadata."""
        with patch("ols.utils.mcp_utils.MultiServerMCPClient") as mock_client_cls:
            tool = MagicMock(spec=StructuredTool)
            tool.name = "test_tool"
            tool.metadata = None

            mock_client = AsyncMock()
            mock_client.get_tools.return_value = [tool]
            mock_client_cls.return_value = mock_client

            servers = {
                "test-server": {"transport": "streamable_http", "url": "http://test"}
            }
            result = await gather_mcp_tools(servers)

            assert len(result) == 1
            assert result[0].metadata is not None
            assert result[0].metadata["mcp_server"] == "test-server"


@pytest.mark.asyncio
class TestGetMcpTools:
    """Tests for get_mcp_tools function."""

    async def test_without_tools_rag(self, mock_file_server, mock_tool):
        """Test getting tools when tools_rag not configured."""
        with (
            patch("ols.utils.mcp_utils.config") as mock_config,
            patch("ols.utils.mcp_utils.gather_mcp_tools") as mock_gather,
        ):
            mock_config.tools_rag = None
            mock_config.mcp_servers.servers = [mock_file_server]
            mock_gather.return_value = [mock_tool]

            result = await get_mcp_tools("test query")

            assert len(result) == 1
            assert result[0].name == "test_tool"

    async def test_with_tools_rag_first_call(self, mock_k8s_server, mock_tool):
        """Test first call with tools_rag (cold start)."""
        with (
            patch("ols.utils.mcp_utils.config") as mock_config,
            patch("ols.utils.mcp_utils.gather_mcp_tools") as mock_gather,
        ):
            # Setup config
            mock_config.tools_rag = MagicMock()
            mock_config.tools_rag.populate_tools = MagicMock()
            mock_config.tools_rag.set_default_servers = MagicMock()
            mock_config.tools_rag.retrieve_hybrid.return_value = {
                "k8s-server": [{"name": "test_tool", "description": "test"}]
            }
            mock_config.k8s_tools_resolved = False
            mock_config.mcp_servers.servers = [mock_k8s_server]
            mock_config.mcp_servers_dict = {"k8s-server": mock_k8s_server}

            mock_gather.return_value = [mock_tool]

            result = await get_mcp_tools(
                "test query", user_token="k8s-token"  # noqa: S106
            )

            # Should populate tools_rag on first call
            assert mock_config.tools_rag.populate_tools.called
            assert len(result) == 1

    async def test_with_client_headers(
        self, mock_k8s_server, mock_client_server, mock_tool
    ):
        """Test with client headers provided."""
        with (
            patch("ols.utils.mcp_utils.config") as mock_config,
            patch("ols.utils.mcp_utils.gather_mcp_tools") as mock_gather,
        ):
            # Setup config
            mock_config.tools_rag = MagicMock()
            mock_config.tools_rag.populate_tools = MagicMock()
            mock_config.tools_rag.set_default_servers = MagicMock()
            mock_config.tools_rag.retrieve_hybrid.return_value = {
                "client-server": [{"name": "test_tool", "description": "test"}]
            }
            mock_config.k8s_tools_resolved = True
            mock_config.mcp_servers.servers = [mock_k8s_server, mock_client_server]
            mock_config.mcp_servers_dict = {
                "k8s-server": mock_k8s_server,
                "client-server": mock_client_server,
            }

            mock_gather.return_value = [mock_tool]

            client_headers = {"client-server": {"Authorization": "Bearer token"}}
            result = await get_mcp_tools(
                "test query",
                user_token="k8s-token",  # noqa: S106
                client_headers=client_headers,
            )

            # Should call retrieve_hybrid with client server names
            mock_config.tools_rag.retrieve_hybrid.assert_called_once()
            call_args = mock_config.tools_rag.retrieve_hybrid.call_args
            assert call_args[0][0] == "test query"  # query argument
            assert call_args[1]["client_servers"] == [
                "client-server"
            ]  # client_servers kwarg
            assert len(result) == 1

    async def test_rag_filtering_error_fallback(self, mock_file_server, mock_tool):
        """Test fallback to all tools when RAG filtering fails."""
        with (
            patch("ols.utils.mcp_utils.config") as mock_config,
            patch("ols.utils.mcp_utils.gather_mcp_tools") as mock_gather,
        ):
            # Setup config
            mock_config.tools_rag = MagicMock()
            mock_config.tools_rag.retrieve_hybrid.side_effect = Exception("RAG error")
            mock_config.k8s_tools_resolved = True
            mock_config.mcp_servers.servers = [mock_file_server]
            mock_config.mcp_servers_dict = {"file-server": mock_file_server}

            mock_gather.return_value = [mock_tool]

            result = await get_mcp_tools("test query")

            # Should fallback to all tools
            assert len(result) == 1
            assert result[0].name == "test_tool"

    async def test_no_matching_tools(self, mock_k8s_server):
        """Test when RAG filtering returns no matches."""
        with patch("ols.utils.mcp_utils.config") as mock_config:
            # Setup config
            mock_config.tools_rag = MagicMock()
            mock_config.tools_rag.retrieve_hybrid.return_value = {}
            mock_config.k8s_tools_resolved = True
            mock_config.mcp_servers.servers = [mock_k8s_server]
            mock_config.mcp_servers_dict = {"k8s-server": mock_k8s_server}

            result = await get_mcp_tools(
                "test query", user_token="k8s-token"  # noqa: S106
            )

            # Should return empty list
            assert result == []

    async def test_no_servers_configured(self):
        """Test when no MCP servers are configured."""
        with (
            patch("ols.utils.mcp_utils.config") as mock_config,
            patch("ols.utils.mcp_utils.gather_mcp_tools") as mock_gather,
        ):
            mock_config.tools_rag = None
            mock_config.mcp_servers.servers = []
            mock_gather.return_value = []

            result = await get_mcp_tools("test query")

            assert result == []
