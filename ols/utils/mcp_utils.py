"""Utilities for parsing and validating MCP client headers."""

import logging
from typing import Any, Optional

from langchain_mcp_adapters.client import MultiServerMCPClient

from ols import constants

logger = logging.getLogger(__name__)


def get_servers_requiring_client_headers(mcp_servers: Any) -> dict[str, list[str]]:
    """Get list of MCP servers that require client-provided headers.

    Args:
        mcp_servers: MCPServers configuration object

    Returns:
        Dictionary mapping server names to lists of required header names

    Examples:
        >>> servers = get_servers_requiring_client_headers(config.mcp_servers)
        >>> servers
        {'github-mcp': ['Authorization'], 'slack-mcp': ['Authorization', 'X-Slack-Team']}
    """
    result: dict[str, list[str]] = {}

    if not mcp_servers or not mcp_servers.servers:
        return result

    for server in mcp_servers.servers:
        required_headers = []

        if server.headers:
            for header_name, header_value in server.resolved_headers.items():
                if header_value == constants.MCP_CLIENT_PLACEHOLDER:
                    required_headers.append(header_name)

        # Only include servers that need client headers
        if required_headers:
            result[server.name] = required_headers

    return result


def resolve_header_value(
    value: str,
    header_name: str,
    server_name: str,
    user_token: Optional[str],
    client_headers: Optional[dict[str, dict[str, str]]],
) -> Optional[str]:
    """Resolve header value by substituting placeholders.

    Args:
        value: Header value (may be a placeholder or actual value)
        header_name: Name of the header
        server_name: Name of the MCP server (for logging)
        user_token: User's kubernetes token (if available)
        client_headers: Client-provided headers (if available)

    Returns:
        Resolved header value, or None if resolution failed
    """
    match value:
        case constants.MCP_KUBERNETES_PLACEHOLDER:
            # Replace "kubernetes" with actual k8s token
            if user_token:
                return f"Bearer {user_token}"
            logger.warning(
                "MCP server %s requires kubernetes token but none available",
                server_name,
            )
            return None

        case constants.MCP_CLIENT_PLACEHOLDER:
            # Replace "client" with value from client headers
            if client_headers and server_name in client_headers:
                server_headers = client_headers[server_name]
                if header_name in server_headers:
                    return server_headers[header_name]

                # Header name not found in client headers
                logger.warning(
                    "MCP server %s requires client header '%s' but not provided",
                    server_name,
                    header_name,
                )
                return None
            logger.warning(
                "MCP server %s requires client headers but none provided",
                server_name,
            )
            return None

        case _:
            # Already resolved (from file) at config load time
            return value


def resolve_server_headers(
    server: Any,
    user_token: Optional[str],
    client_headers: Optional[dict[str, dict[str, str]]],
) -> dict[str, str] | None:
    """Resolve headers for a single MCP server by replacing placeholders.

    Args:
        server: MCP server configuration
        user_token: User's kubernetes token (if available)
        client_headers: Client-provided headers (if available)

    Returns:
        Resolved headers dict, or None if resolution failed
    """
    headers = {}

    # Loop through configured headers and replace placeholders
    for header_name, header_value in server.resolved_headers.items():
        resolved_value = resolve_header_value(
            header_value, header_name, server.name, user_token, client_headers
        )

        if resolved_value is None:
            # Resolution failed (logged in resolve_header_value)
            return None

        headers[header_name] = resolved_value

    return headers


async def gather_mcp_tools(mcp_servers: dict[str, Any]) -> list:
    """Gather tools from multiple MCP servers with failure isolation.

    Load tools from each MCP server individually so that if one server
    is unreachable, tools from other servers are still available.

    Args:
        mcp_servers: Dictionary mapping server names to their configurations.

    Returns:
        List of tools from all successfully connected servers.
    """
    all_tools: list = []
    mcp_client = MultiServerMCPClient(mcp_servers)

    for server_name in mcp_servers:
        try:
            server_tools = await mcp_client.get_tools(server_name=server_name)
            all_tools.extend(server_tools)
            logger.info(
                "Loaded %d tools from MCP server '%s'",
                len(server_tools),
                server_name,
            )
        except Exception as e:
            logger.error("Failed to get tools from MCP server '%s': %s", server_name, e)

    return all_tools


def build_mcp_config(
    mcp_servers_config: Any,
    user_token: Optional[str],
    client_headers: Optional[dict[str, dict[str, str]]],
) -> dict[str, Any]:
    """Build MCP client configuration from config.

    Resolves authorization headers, substituting placeholders with runtime values
    (e.g., "kubernetes" → user token).

    Args:
        mcp_servers_config: MCPServers configuration object
        user_token: User's kubernetes token (if available)
        client_headers: Client-provided headers (if available)

    Returns:
        Dictionary mapping server names to their config for MultiServerMCPClient.
        Returns empty dict if no MCP servers configured or on error.
    """
    if not mcp_servers_config or not mcp_servers_config.servers:
        return {}

    servers_config: dict[str, Any] = {}

    try:
        for server in mcp_servers_config.servers:
            headers = resolve_server_headers(server, user_token, client_headers)
            if headers is None:
                continue

            # Build MultiServerMCPClient config format
            servers_config[server.name] = {
                "transport": "streamable_http",
                "url": server.url,
            }
            if headers:
                servers_config[server.name]["headers"] = headers
            if server.timeout:
                servers_config[server.name]["timeout"] = server.timeout

    except Exception as e:
        logger.error("Failed to build MCP config: %s", e)
        return {}

    return servers_config
