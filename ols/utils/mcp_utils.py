"""Utilities for parsing and validating MCP client headers."""

import logging
from typing import Optional, TypeAlias, TypedDict

from langchain_core.tools.structured import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from ols import config, constants
from ols.app.models.config import MCPServerConfig, MCPServers

logger = logging.getLogger(__name__)


class MCPServerTransport(TypedDict, total=False):
    """Type definition for MCP server transport configuration."""

    transport: str
    url: str
    headers: dict[str, str]
    timeout: int


# Type aliases for clarity and reusability
ClientHeaders: TypeAlias = dict[str, dict[str, str]]
MCPServersDict: TypeAlias = dict[str, MCPServerTransport]


def get_servers_requiring_client_headers(
    mcp_servers: MCPServers | None,
) -> dict[str, list[str]]:
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
    client_headers: ClientHeaders | None,
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
    server: MCPServerConfig,
    user_token: Optional[str],
    client_headers: ClientHeaders | None,
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


def _normalize_tool_schema(tool: StructuredTool) -> None:
    """Normalize an MCP tool's dict schema for OpenAI compatibility.

    Some MCP tools accept no arguments, producing a valid JSON Schema like
    ``{"type": "object"}`` with no ``properties`` key. This causes:

    1. LangChain's ``BaseTool.args`` raises ``KeyError("properties")``.
    2. OpenAI rejects the function with *"object schema missing properties"*.

    Only dict schemas with ``"type": "object"`` are patched. Pydantic model
    schemas always include ``properties`` and are left untouched.
    """
    schema = tool.args_schema
    if not isinstance(schema, dict):
        return
    if schema.get("type") != "object":
        return
    schema.setdefault("properties", {})
    schema.setdefault("required", [])


async def gather_mcp_tools(
    mcp_servers: MCPServersDict, allowed_tool_names: Optional[set[str]] = None
) -> list[StructuredTool]:
    """Gather tools from multiple MCP servers with failure isolation.

    Load tools from each MCP server individually so that if one server
    is unreachable, tools from other servers are still available.

    Args:
        mcp_servers: Dictionary mapping server names to their configurations.
        allowed_tool_names: Optional set of tool names to filter by. If provided,
            only tools with names in this set will be included.

    Returns:
        List of tools from all successfully connected servers.
        Each tool has metadata indicating which MCP server it came from.
    """
    all_tools: list[StructuredTool] = []
    mcp_client = MultiServerMCPClient(mcp_servers)

    for server_name in mcp_servers:
        try:
            server_tools = await mcp_client.get_tools(server_name=server_name)

            # Filter immediately if we have an allowlist
            if allowed_tool_names:
                server_tools = [
                    tool for tool in server_tools if tool.name in allowed_tool_names
                ]

            # Add MCP server name to each tool's metadata
            for tool in server_tools:
                _normalize_tool_schema(tool)
                if not hasattr(tool, "metadata") or tool.metadata is None:
                    tool.metadata = {}
                tool.metadata["mcp_server"] = server_name

            all_tools.extend(server_tools)
            logger.info(
                "Loaded %d tools from MCP server '%s'",
                len(server_tools),
                server_name,
            )
        except Exception as e:
            logger.error("Failed to get tools from MCP server '%s': %s", server_name, e)

    return all_tools


async def _gather_and_populate_tools(
    servers_list: list[MCPServerConfig],
    user_token: Optional[str],
    client_headers: ClientHeaders | None,
    populate_to_rag: bool = False,
    allowed_tool_names: Optional[set[str]] = None,
    deduplicate: bool = False,
) -> tuple[MCPServersDict, list[StructuredTool]]:
    """Build MCP config and gather tools, optionally populating ToolsRAG.

    Args:
        servers_list: List of MCPServerConfig objects
        user_token: Optional user authentication token
        client_headers: Optional client-provided headers
        populate_to_rag: If True, add gathered tools to ToolsRAG
        allowed_tool_names: Optional set of tool names to filter by
        deduplicate: If True, deduplicate tools by name (first-seen wins)

    Returns:
        Tuple of (servers_config dict, tools list)
    """
    servers_config = build_mcp_config(servers_list, user_token, client_headers)

    if not servers_config:
        return {}, []

    tools = await gather_mcp_tools(servers_config, allowed_tool_names)

    if tools and populate_to_rag and config.tools_rag:
        config.tools_rag.populate_tools(tools)

    if deduplicate:
        seen_names: set[str] = set()
        unique_tools: list[StructuredTool] = []
        for tool in tools:
            if tool.name not in seen_names:
                seen_names.add(tool.name)
                unique_tools.append(tool)
            else:
                logger.warning(
                    "Duplicate tool '%s' from server '%s' skipped",
                    tool.name,
                    (tool.metadata or {}).get("mcp_server", "unknown"),
                )
        tools = unique_tools

    return servers_config, tools


async def _populate_tools_rag(
    user_token: Optional[str],
    client_headers: ClientHeaders | None,
) -> None:
    """Populate ToolsRAG with tools from k8s-auth and client-auth MCP servers.

    Args:
        user_token: Optional user authentication token
        client_headers: Optional client-provided MCP headers
    """
    if not config.k8s_tools_resolved:
        k8s_servers_config, k8s_tools = await _gather_and_populate_tools(
            config.mcp_servers.servers,
            user_token=user_token,
            client_headers=None,
            populate_to_rag=True,
        )

        if k8s_tools:
            logger.info(
                "Populated ToolsRAG with %d tools from %d k8s-auth MCP servers",
                len(k8s_tools),
                len(k8s_servers_config),
            )
            config.tools_rag.set_default_servers(list(k8s_servers_config.keys()))

        config.k8s_tools_resolved = True

    if client_headers:
        client_servers_list = [
            config.mcp_servers_dict[name]
            for name in client_headers.keys()
            if name in config.mcp_servers_dict
        ]

        if client_servers_list:
            client_servers_config, client_tools = await _gather_and_populate_tools(
                client_servers_list,
                user_token,
                client_headers,
                populate_to_rag=True,
            )

            if client_tools:
                logger.info(
                    "Added %d tools from %d client-auth MCP servers to ToolsRAG",
                    len(client_tools),
                    len(client_servers_config),
                )


async def get_mcp_tools(
    query: str,
    user_token: Optional[str] = None,
    client_headers: ClientHeaders | None = None,
) -> list[StructuredTool]:
    """Get all MCP tools, handling tools_rag population if configured.

    Args:
        query: The user's query for filtering tools
        user_token: Optional user authentication token for tool access
        client_headers: Optional client-provided MCP headers for authentication

    Returns:
        List of all tools from MCP servers (filtered if tools_rag configured).
    """
    # If tools_rag is not configured, return all tools
    if not config.tools_rag:
        mcp_servers_config, all_tools = await _gather_and_populate_tools(
            config.mcp_servers.servers, user_token, client_headers, deduplicate=True
        )

        if not mcp_servers_config:
            logger.debug("No MCP servers provided, tool calling is disabled")
            return []

        logger.info("MCP servers provided: %s", list(mcp_servers_config.keys()))
        return all_tools

    await _populate_tools_rag(user_token, client_headers)

    # Filter tools using ToolsRAG with server filtering
    try:
        # Build list of client server names if provided
        client_server_names = list(client_headers.keys()) if client_headers else None

        # Query with client servers (combined with defaults internally)
        filtered_result = config.tools_rag.retrieve_hybrid(
            query, client_servers=client_server_names
        )
    except Exception as e:
        logger.error(
            "Failed to filter tools using ToolsRAG: %s, falling back to all tools",
            e,
        )
        # Fallback: return all tools unfiltered rather than empty list
        _, all_tools = await _gather_and_populate_tools(
            config.mcp_servers.servers, user_token, client_headers, deduplicate=True
        )
        return all_tools

    # Extract tool names and server names from filtered results
    # filtered_result is dict[server_name, list[tool_dicts]]
    if filtered_result:
        # Collect all tool names and server names from the grouped results
        filtered_tool_names: set[str] = set()
        server_names: set[str] = set()

        for server_name, tools_list in filtered_result.items():
            server_names.add(server_name)
            for tool in tools_list:
                # Tool dict has 'name' field
                if "name" in tool:
                    filtered_tool_names.add(tool["name"])

        # Get server configs directly using dict lookup
        filtered_servers_list = [
            config.mcp_servers_dict[name]
            for name in server_names
            if name in config.mcp_servers_dict
        ]

        if not filtered_servers_list:
            logger.warning(
                "No matching servers found in config for filtered tools. "
                "Filtered tools referenced servers: %s",
                server_names,
            )
            return []

        # Build config for only the filtered servers and gather tools
        filtered_servers_config, filtered_tools = await _gather_and_populate_tools(
            filtered_servers_list,
            user_token,
            client_headers,
            allowed_tool_names=filtered_tool_names,
            deduplicate=True,
        )

        logger.info(
            "Filtered to %d tools from %d MCP servers based on query",
            len(filtered_tools),
            len(filtered_servers_config),
        )
        return filtered_tools

    # Fallback: return empty list if filtering failed
    logger.warning("No tools matched the query filter")
    return []


def build_mcp_config(
    servers_list: list[MCPServerConfig],
    user_token: Optional[str],
    client_headers: ClientHeaders | None,
) -> MCPServersDict:
    """Build MCP client configuration from config.

    Resolves authorization headers, substituting placeholders with runtime values
    (e.g., "kubernetes" → user token).

    Args:
        servers_list: List of MCPServerConfig objects
        user_token: User's kubernetes token (if available)
        client_headers: Client-provided headers (if available)

    Returns:
        Dictionary mapping server names to their config for MultiServerMCPClient.
        Returns empty dict if no MCP servers configured or on error.
    """
    if not servers_list:
        return {}

    servers_config: MCPServersDict = {}

    try:
        for server in servers_list:
            headers = resolve_server_headers(server, user_token, client_headers)
            if headers is None:
                continue

            # Build MultiServerMCPClient config format
            server_config: MCPServerTransport = {
                "transport": "streamable_http",
                "url": server.url,
            }
            if headers:
                server_config["headers"] = headers
            if server.timeout:
                server_config["timeout"] = server.timeout

            servers_config[server.name] = server_config

    except Exception as e:
        logger.error("Failed to build MCP config: %s", e)
        return {}

    return servers_config
