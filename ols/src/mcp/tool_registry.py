"""MCP tool metadata registry for tracking UI-enabled tools."""

import logging
from dataclasses import dataclass
from typing import Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from ols import config

logger = logging.getLogger(__name__)


@dataclass
class ToolUIMetadata:
    """Metadata for a tool's UI capabilities.

    Tools may have a resource_uri (for rendering UI), visibility constraints,
    or both. A tool with visibility=["app"] and no resource_uri is an app-only
    tool that the iframe calls directly but that has no standalone UI resource.
    """

    server_name: str
    resource_uri: Optional[str] = None
    visibility: Optional[list[str]] = None


_tool_ui_registry: dict[str, ToolUIMetadata] = {}
_resource_uri_to_config_name: dict[str, str] = {}


def get_tool_ui_metadata(tool_name: str) -> Optional[ToolUIMetadata]:
    """Get UI metadata for a tool if it has MCP App support.

    Args:
        tool_name: The name of the tool.

    Returns:
        ToolUIMetadata if the tool has UI support, None otherwise.
    """
    return _tool_ui_registry.get(tool_name)


def is_model_visible(tool_name: str) -> bool:
    """Check whether a tool should be visible to the LLM.

    Tools with _meta.ui.visibility that does not include "model" are app-only
    and should not be bound to the LLM. Tools without UI metadata or without
    an explicit visibility constraint default to model-visible.

    Args:
        tool_name: The name of the tool.

    Returns:
        True if the tool should be included in LLM tool binding.
    """
    ui_meta = _tool_ui_registry.get(tool_name)
    if not ui_meta or not ui_meta.visibility:
        return True
    return "model" in ui_meta.visibility


def get_config_name_for_resource_uri(resource_uri: str) -> Optional[str]:
    """Get the config server name for a ui:// resource URI.

    The full URI is treated as an opaque identifier; no internal structure
    (authority, path) is assumed.

    Args:
        resource_uri: The full ui:// resource URI.

    Returns:
        The config server name if the URI was discovered, None otherwise.
    """
    return _resource_uri_to_config_name.get(resource_uri)


def register_tool_ui(
    tool_name: str,
    server_name: str,
    resource_uri: Optional[str] = None,
    visibility: Optional[list[str]] = None,
) -> None:
    """Register a tool's UI metadata.

    Args:
        tool_name: The name of the tool.
        server_name: The MCP server name.
        resource_uri: The UI resource URI (None for visibility-only tools).
        visibility: Tool visibility scope (e.g. ["model"], ["app"], ["model", "app"]).
    """
    _tool_ui_registry[tool_name] = ToolUIMetadata(
        server_name=server_name,
        resource_uri=resource_uri,
        visibility=visibility,
    )
    logger.info(
        "Registered UI for tool '%s' from server '%s' (visibility=%s)",
        tool_name,
        server_name,
        visibility,
    )


async def discover_tool_ui_metadata() -> None:
    """Discover and register UI metadata for all configured MCP servers.

    This queries each MCP server for its tools and checks for _meta.ui metadata.
    """
    if not config.mcp_servers or not config.mcp_servers.servers:
        logger.debug("No MCP servers configured, skipping UI metadata discovery")
        return

    for server in config.mcp_servers.servers:
        try:
            await _discover_server_tools(server.name, server.url, server.headers)
        except Exception as e:
            logger.warning(
                "Failed to discover UI metadata from MCP server '%s': %s",
                server.name,
                e,
            )


async def _discover_server_tools(
    server_name: str,
    server_url: str,
    headers: Optional[dict[str, str]] = None,
) -> None:
    """Discover tools with UI metadata from a single MCP server.

    Args:
        server_name: The name of the MCP server.
        server_url: The URL of the MCP server.
        headers: Optional authentication headers.
    """
    logger.debug(
        "Discovering UI metadata from MCP server '%s' at %s", server_name, server_url
    )

    resolved_headers = {}
    if headers:
        for key, value in headers.items():
            if value.startswith("/"):
                try:
                    with open(value) as f:
                        resolved_headers[key] = f.read().strip()
                except Exception:
                    resolved_headers[key] = value
            elif value in ("kubernetes", "client"):
                continue
            else:
                resolved_headers[key] = value

    try:
        async with streamablehttp_client(
            url=server_url,
            headers=resolved_headers,
            timeout=30,
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                tools_result = await session.list_tools()

                for tool in tools_result.tools:
                    # MCP SDK uses 'meta' as Python attr (aliased from '_meta' in JSON)
                    meta = getattr(tool, "meta", None) or {}

                    # Check both formats: _meta.ui.resourceUri and _meta["ui/resourceUri"]
                    ui_meta = meta.get("ui", {}) if isinstance(meta, dict) else {}
                    resource_uri = ui_meta.get("resourceUri")
                    if not resource_uri and isinstance(meta, dict):
                        resource_uri = meta.get("ui/resourceUri")

                    visibility = ui_meta.get("visibility") if isinstance(ui_meta, dict) else None

                    has_ui_metadata = resource_uri or visibility

                    if has_ui_metadata:
                        register_tool_ui(
                            tool.name, server_name, resource_uri, visibility=visibility,
                        )

                        if resource_uri and resource_uri.startswith("ui://"):
                            existing = _resource_uri_to_config_name.get(resource_uri)
                            if existing and existing != server_name:
                                logger.error(
                                    "Resource URI '%s' already mapped to server '%s', "
                                    "overwriting with '%s' - one server's UI resources "
                                    "will be unreachable",
                                    resource_uri,
                                    existing,
                                    server_name,
                                )
                            _resource_uri_to_config_name[resource_uri] = server_name
                            logger.info(
                                "Mapped resource URI '%s' to config server '%s'",
                                resource_uri,
                                server_name,
                            )

                        logger.debug(
                            "Found UI metadata for tool '%s' (resource=%s, visibility=%s)",
                            tool.name,
                            resource_uri,
                            visibility,
                        )
                    else:
                        logger.debug(
                            "Tool '%s' has no UI metadata (meta=%s)",
                            tool.name,
                            meta,
                        )

    except Exception as e:
        logger.warning("Error connecting to MCP server '%s': %s", server_name, e)
        raise
