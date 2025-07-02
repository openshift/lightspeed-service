"""Tests for MCPConfigBuilder."""

import os
from datetime import timedelta
from unittest.mock import patch

from ols.app.models.config import (
    MCPServerConfig,
    SseTransportConfig,
    StdioTransportConfig,
    StreamableHttpTransportConfig,
)
from ols.src.tools.mcp_config_builder import K8S_AUTH_HEADER, MCPConfigBuilder


def test_mcp_config_builder_dump_client_config():
    """Test MCPConfigBuilder.dump_client_config method."""
    mcp_server_configs = [
        MCPServerConfig(
            name="openshift",
            transport="stdio",
            stdio=StdioTransportConfig(
                command="hello",
                env={"X": "Y"},
            ),
        ),
        MCPServerConfig(
            name="not-openshift",
            transport="stdio",
            stdio=StdioTransportConfig(
                command="hello",
                env={"X": "Y"},
            ),
        ),
    ]
    user_token = "fake-token"  # noqa: S105

    # patch the environment variable to avoid using values from the system
    with patch.dict(os.environ, {}, clear=True):
        builder = MCPConfigBuilder(user_token, mcp_server_configs)
        mcp_config = builder.dump_client_config()

    assert mcp_config == {
        "openshift": {
            "transport": "stdio",
            "command": "hello",
            "args": [],
            "env": {"X": "Y", "OC_USER_TOKEN": "fake-token"},
            "cwd": ".",
            "encoding": "utf-8",
        },
        "not-openshift": {
            "transport": "stdio",
            "command": "hello",
            "args": [],
            "env": {"X": "Y"},
            "cwd": ".",
            "encoding": "utf-8",
        },
    }


class TestMCPConfigBuilder:
    """Test MCPConfigBuilder class."""

    @staticmethod
    def test_include_auth_to_stdio():
        """Test include_auth_to_stdio method."""
        user_token = "fake-token"  # noqa: S105
        envs = {"A": 42, "KUBECONFIG": "bla"}

        builder = MCPConfigBuilder(user_token, [])
        mcp_config = builder.include_auth_to_stdio(envs)

        expected = {**envs, "OC_USER_TOKEN": user_token}
        assert mcp_config == expected

    @staticmethod
    def test_token_set_in_env(caplog):
        """Test include_auth_to_stdio with token set in env."""
        # OC_USER_TOKEN set in env - is logged and overriden
        user_token = "fake-token"  # noqa: S105
        envs = {"OC_USER_TOKEN": "different-value"}

        builder = MCPConfigBuilder(user_token, [])
        with patch.dict(os.environ, {}, clear=True):
            mcp_config = builder.include_auth_to_stdio(envs)

        expected = {"OC_USER_TOKEN": user_token}
        assert mcp_config == expected
        assert "overriding with actual user token" in caplog.text

    @staticmethod
    def test_kubeconfig_from_environ(caplog):
        """Test include_auth_to_stdio with KUBECONFIG from environment."""
        # KUBECONFIG is not set in env - value from os.environ is used
        caplog.set_level(20)  # info
        envs = {"A": 42}
        user_token = "fake-token"  # noqa: S105

        builder = MCPConfigBuilder(user_token, [])
        with patch.dict(os.environ, {"KUBECONFIG": "os value"}):
            mcp_config = builder.include_auth_to_stdio(envs)

        expected = {**envs, "OC_USER_TOKEN": user_token, "KUBECONFIG": "os value"}
        assert mcp_config == expected
        assert "Using KUBECONFIG from environment" in caplog.text

    @staticmethod
    def test_kubernetes_service_from_environ(caplog):
        """Test include_auth_to_stdio with KUBERNETES_SERVICE_* from environment."""
        # KUBECONFIG is not set, but KUBERNETES_SERVICE_* is available
        caplog.set_level(20)  # info
        envs = {"A": 42}
        user_token = "fake-token"  # noqa: S105

        builder = MCPConfigBuilder(user_token, [])
        with patch.dict(
            os.environ,
            {"KUBERNETES_SERVICE_HOST": "k8s-host", "KUBERNETES_SERVICE_PORT": "8443"},
        ):
            mcp_config = builder.include_auth_to_stdio(envs)

        expected = {
            **envs,
            "OC_USER_TOKEN": user_token,
            "KUBERNETES_SERVICE_HOST": "k8s-host",
            "KUBERNETES_SERVICE_PORT": "8443",
        }
        assert mcp_config == expected
        assert "Using KUBERNETES_SERVICE_* from environment" in caplog.text

    @staticmethod
    def test_missing_kubeconfig_and_kubernetes_service(caplog):
        """Test include_auth_to_stdio with missing KUBECONFIG and KUBERNETES_SERVICE_*."""
        # Both KUBECONFIG and KUBERNETES_SERVICE_* are missing
        envs = {}
        user_token = "fake-token"  # noqa: S105

        builder = MCPConfigBuilder(user_token, [])
        with patch.dict(os.environ, {}, clear=True):
            mcp_config = builder.include_auth_to_stdio(envs)

        expected = {"OC_USER_TOKEN": user_token}
        assert mcp_config == expected
        assert "Missing necessary KUBECONFIG/KUBERNETES_SERVICE_* envs" in caplog.text

    @staticmethod
    def test_include_auth_header():
        """Test include_auth_header method."""
        user_token = "fake-token"  # noqa: S105
        config = {}

        result = MCPConfigBuilder.include_auth_header(user_token, config)

        assert "headers" in result
        assert result["headers"][K8S_AUTH_HEADER] == f"Bearer {user_token}"

    @staticmethod
    def test_include_auth_header_existing_headers():
        """Test include_auth_header with existing headers."""
        user_token = "fake-token"  # noqa: S105
        config = {"headers": {"Content-Type": "application/json"}}

        result = MCPConfigBuilder.include_auth_header(user_token, config)

        assert result["headers"]["Content-Type"] == "application/json"
        assert result["headers"][K8S_AUTH_HEADER] == f"Bearer {user_token}"

    @staticmethod
    def test_include_auth_header_existing_auth(caplog):
        """Test include_auth_header with existing auth header."""
        user_token = "fake-token"  # noqa: S105
        config = {"headers": {K8S_AUTH_HEADER: "old-token"}}

        result = MCPConfigBuilder.include_auth_header(user_token, config)

        assert result["headers"][K8S_AUTH_HEADER] == f"Bearer {user_token}"
        assert (
            "Kubernetes auth header is already set, overriding with actual user token"
            in caplog.text
        )

    @staticmethod
    def test_dump_client_config_with_sse():
        """Test dump_client_config with SSE configuration."""
        mcp_server_configs = [
            MCPServerConfig(
                name="sse-server",
                transport="sse",
                sse=SseTransportConfig(
                    url="https://example.com/events",
                    headers={"X-Custom-Header": "value"},
                ),
            ),
        ]
        user_token = "fake-token"  # noqa: S105

        builder = MCPConfigBuilder(user_token, mcp_server_configs)
        result = builder.dump_client_config()

        assert result["sse-server"]["transport"] == "sse"
        assert result["sse-server"]["url"] == "https://example.com/events"
        assert result["sse-server"]["headers"]["X-Custom-Header"] == "value"
        assert (
            result["sse-server"]["headers"][K8S_AUTH_HEADER] == f"Bearer {user_token}"
        )

    @staticmethod
    def test_dump_client_config_with_mixed_transports():
        """Test dump_client_config with both SSE and stdio configurations."""
        mcp_server_configs = [
            MCPServerConfig(
                name="openshift",
                transport="stdio",
                stdio=StdioTransportConfig(
                    command="hello",
                    env={"X": "Y"},
                ),
            ),
            MCPServerConfig(
                name="sse-server",
                transport="sse",
                sse=SseTransportConfig(
                    url="https://example.com/events",
                ),
            ),
        ]
        user_token = "fake-token"  # noqa: S105

        with patch.dict(os.environ, {}, clear=True):
            builder = MCPConfigBuilder(user_token, mcp_server_configs)
            result = builder.dump_client_config()

        assert "openshift" in result
        assert "sse-server" in result
        assert result["openshift"]["transport"] == "stdio"
        assert result["sse-server"]["transport"] == "sse"
        assert result["openshift"]["env"]["OC_USER_TOKEN"] == user_token
        assert (
            result["sse-server"]["headers"][K8S_AUTH_HEADER] == f"Bearer {user_token}"
        )

    @staticmethod
    def test_dump_client_config_with_streamable_http():
        """Test dump_client_config with streamable HTTP configuration."""
        mcp_server_configs = [
            MCPServerConfig(
                name="streamable-server",
                transport="streamable_http",
                streamable_http=StreamableHttpTransportConfig(
                    url="https://example.com/stream",
                    headers={"X-Custom-Header": "value"},
                    timeout=30,
                    sse_read_timeout=60,
                ),
            ),
        ]
        user_token = "fake-token"  # noqa: S105

        builder = MCPConfigBuilder(user_token, mcp_server_configs)
        result = builder.dump_client_config()

        assert result["streamable-server"]["transport"] == "streamable_http"
        assert result["streamable-server"]["url"] == "https://example.com/stream"
        assert result["streamable-server"]["headers"]["X-Custom-Header"] == "value"
        assert (
            result["streamable-server"]["headers"][K8S_AUTH_HEADER]
            == f"Bearer {user_token}"
        )
        # Verify that timeout values are converted to timedelta objects
        assert result["streamable-server"]["timeout"] == timedelta(seconds=30)
        assert result["streamable-server"]["sse_read_timeout"] == timedelta(seconds=60)

    @staticmethod
    def test_dump_client_config_with_all_transports():
        """Test dump_client_config with stdio, SSE, and streamable HTTP configurations."""
        mcp_server_configs = [
            MCPServerConfig(
                name="openshift",
                transport="stdio",
                stdio=StdioTransportConfig(
                    command="hello",
                    env={"X": "Y"},
                ),
            ),
            MCPServerConfig(
                name="sse-server",
                transport="sse",
                sse=SseTransportConfig(
                    url="https://example.com/events",
                ),
            ),
            MCPServerConfig(
                name="streamable-server",
                transport="streamable_http",
                streamable_http=StreamableHttpTransportConfig(
                    url="https://example.com/stream",
                ),
            ),
        ]
        user_token = "fake-token"  # noqa: S105

        with patch.dict(os.environ, {}, clear=True):
            builder = MCPConfigBuilder(user_token, mcp_server_configs)
            result = builder.dump_client_config()

        assert "openshift" in result
        assert "sse-server" in result
        assert "streamable-server" in result
        assert result["openshift"]["transport"] == "stdio"
        assert result["sse-server"]["transport"] == "sse"
        assert result["streamable-server"]["transport"] == "streamable_http"
        assert result["openshift"]["env"]["OC_USER_TOKEN"] == user_token
        assert (
            result["sse-server"]["headers"][K8S_AUTH_HEADER] == f"Bearer {user_token}"
        )
        assert (
            result["streamable-server"]["headers"][K8S_AUTH_HEADER]
            == f"Bearer {user_token}"
        )
