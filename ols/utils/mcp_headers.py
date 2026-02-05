"""Utilities for parsing and validating MCP client headers."""

import json
import logging
from typing import Any

from ols.constants import MCP_CLIENT_PLACEHOLDER

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
                if header_value == MCP_CLIENT_PLACEHOLDER:
                    required_headers.append(header_name)

        # Only include servers that need client headers
        if required_headers:
            result[server.name] = required_headers

    return result


def _validate_header_dict(
    header_obj: Any, server_name: str, idx: int
) -> dict[str, str] | None:
    """Validate a single header dictionary.

    Args:
        header_obj: The header object to validate
        server_name: Name of the server (for logging)
        idx: Index in the list (for logging)

    Returns:
        Validated header dict, or None if invalid
    """
    if not isinstance(header_obj, dict):
        logger.warning(
            "MCP-Headers for server '%s' index %d must be an object, got %s",
            server_name,
            idx,
            type(header_obj).__name__,
        )
        return None

    validated_header: dict[str, str] = {}
    for key, value in header_obj.items():
        if not isinstance(key, str):
            logger.warning(
                "MCP-Headers key for server '%s' must be string, skipping: %s",
                server_name,
                key,
            )
            continue

        if not isinstance(value, str):
            logger.warning(
                "MCP-Headers value for server '%s' key '%s' must be string, got %s",
                server_name,
                key,
                type(value).__name__,
            )
            continue

        validated_header[key] = value

    return validated_header if validated_header else None


def _validate_server_headers(
    server_headers: Any, server_name: str
) -> list[dict[str, str]] | None:
    """Validate headers list for a single server.

    Args:
        server_headers: The headers value to validate
        server_name: Name of the server (for logging)

    Returns:
        List of validated header dicts, or None if invalid
    """
    if not isinstance(server_headers, list):
        logger.warning(
            "MCP-Headers for server '%s' must be a list, got %s",
            server_name,
            type(server_headers).__name__,
        )
        return None

    validated_headers_list: list[dict[str, str]] = []
    for idx, header_obj in enumerate(server_headers):
        validated_header = _validate_header_dict(header_obj, server_name, idx)
        if validated_header:
            validated_headers_list.append(validated_header)

    return validated_headers_list if validated_headers_list else None


def parse_mcp_headers(header_value: str | None) -> dict[str, list[dict[str, str]]]:
    """Parse MCP-Headers from HTTP request.

    Extracts and validates client-provided headers that should be forwarded
    to MCP servers. The header value should be a JSON object mapping server
    names to a list of header objects. Multiple header objects allow a single
    server to use multiple sets of headers.

    Args:
        header_value: Raw header value from HTTP request, expected to be JSON

    Returns:
        Dictionary mapping server names to lists of header dictionaries,
        empty dict if parsing fails or input is None

    Examples:
        >>> parse_mcp_headers('{"server1": [{"Authorization": "Bearer token123"}]}')
        {'server1': [{'Authorization': 'Bearer token123'}]}

        >>> parse_mcp_headers('{"server1": [{"Auth": "token1"}, {"X-Key": "key123"}]}')
        {'server1': [{'Auth': 'token1'}, {'X-Key': 'key123'}]}

        >>> parse_mcp_headers('invalid json')
        {}

        >>> parse_mcp_headers(None)
        {}
    """
    if not header_value:
        return {}

    try:
        parsed = json.loads(header_value)

        if not isinstance(parsed, dict):
            logger.warning(
                "MCP-Headers must be a JSON object, got %s", type(parsed).__name__
            )
            return {}

        result: dict[str, list[dict[str, str]]] = {}

        for server_name, server_headers in parsed.items():
            if not isinstance(server_name, str):
                logger.warning(
                    "MCP-Headers server name must be string, skipping: %s", server_name
                )
                continue

            validated_headers = _validate_server_headers(server_headers, server_name)
            if validated_headers:
                result[server_name] = validated_headers

        return result

    except json.JSONDecodeError as e:
        logger.warning("Failed to parse MCP-Headers as JSON: %s", e)
        return {}
    except Exception as e:
        logger.error("Unexpected error parsing MCP-Headers: %s", e)
        return {}
