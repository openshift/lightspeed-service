"""Handler for MCP Apps REST API endpoints.

MCP Apps endpoints enable the console to render interactive UIs from MCP servers.
These endpoints proxy resource and tool requests to configured MCP servers.
"""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from ols import config
from ols.app.models.models import (
    ErrorResponse,
    ForbiddenResponse,
    MCPAppResourceRequest,
    MCPAppResourceResponse,
    MCPAppToolCallRequest,
    MCPAppToolCallResponse,
    UnauthorizedResponse,
)
from ols.src.auth.auth import get_auth_dependency
from ols.src.mcp.tool_registry import get_config_name_for_resource_uri

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/mcp-apps", tags=["mcp-apps"])
auth_dependency = get_auth_dependency(config.ols_config, virtual_path="/ols-access")


def get_mcp_server_config(server_name: str) -> dict[str, Any]:
    """Get MCP server configuration by name.

    Args:
        server_name: The name of the MCP server.

    Returns:
        Dictionary containing the server configuration.

    Raises:
        HTTPException: If the server is not found.
    """
    if not config.mcp_servers or not config.mcp_servers.servers:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No MCP servers configured",
        )

    for server in config.mcp_servers.servers:
        if server.name == server_name:
            return {
                "url": server.url,
                "timeout": server.timeout or 30,
                "headers": getattr(server, "_resolved_headers", {}),
            }

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"MCP server '{server_name}' not found",
    )


def validate_resource_uri(resource_uri: str) -> None:
    """Validate that a resource URI uses the ui:// scheme.

    The full URI is treated as an opaque identifier; no internal structure
    (authority, path) is assumed.

    Args:
        resource_uri: The URI to validate.

    Raises:
        HTTPException: If the URI does not use the ui:// scheme.
    """
    if not resource_uri.startswith("ui://") or len(resource_uri) <= 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid resource URI format: {resource_uri}. Expected ui://...",
        )


resource_responses: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "Resource successfully fetched",
        "model": MCPAppResourceResponse,
    },
    400: {
        "description": "Invalid resource URI format",
        "model": ErrorResponse,
    },
    401: {
        "description": "Missing or invalid credentials provided by client",
        "model": UnauthorizedResponse,
    },
    403: {
        "description": "Client does not have permission to access resource",
        "model": ForbiddenResponse,
    },
    404: {
        "description": "MCP server or resource not found",
        "model": ErrorResponse,
    },
    500: {
        "description": "Internal server error",
        "model": ErrorResponse,
    },
}


@router.post("/resources", responses=resource_responses)
async def get_mcp_app_resource(
    request: MCPAppResourceRequest,
    auth: Annotated[Any, Depends(auth_dependency)],
) -> MCPAppResourceResponse:
    """Fetch an MCP App UI resource from an MCP server.

    Args:
        request: The resource request containing the URI.
        auth: The authentication handler.

    Returns:
        The resource content (HTML/JS/CSS).
    """
    logger.debug("MCP Apps resource request: %s", request.resource_uri)

    validate_resource_uri(request.resource_uri)
    config_name = get_config_name_for_resource_uri(request.resource_uri)
    if not config_name:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No MCP server registered for resource URI '{request.resource_uri}'. "
            "Ensure the server is running and was reachable at startup.",
        )
    server_config = get_mcp_server_config(config_name)

    try:
        async with streamablehttp_client(
            url=server_config["url"],
            headers=server_config.get("headers", {}),
            timeout=server_config.get("timeout", 30),
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                result = await session.read_resource(request.resource_uri)

                if not result.contents:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Resource not found: {request.resource_uri}",
                    )

                content = result.contents[0]
                return MCPAppResourceResponse(
                    uri=str(content.uri),
                    mime_type=getattr(content, "mimeType", "text/html"),
                    content=content.text if hasattr(content, "text") else "",
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching MCP resource: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch resource: {e}",
        )


tool_call_responses: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "Tool call successful",
        "model": MCPAppToolCallResponse,
    },
    400: {
        "description": "Invalid request",
        "model": ErrorResponse,
    },
    401: {
        "description": "Missing or invalid credentials provided by client",
        "model": UnauthorizedResponse,
    },
    403: {
        "description": "Client does not have permission to access resource",
        "model": ForbiddenResponse,
    },
    404: {
        "description": "MCP server or tool not found",
        "model": ErrorResponse,
    },
    500: {
        "description": "Internal server error",
        "model": ErrorResponse,
    },
}


@router.post("/tools/call", responses=tool_call_responses)
async def call_mcp_app_tool(
    request: MCPAppToolCallRequest,
    auth: Annotated[Any, Depends(auth_dependency)],
) -> MCPAppToolCallResponse:
    """Call an MCP tool and return the result for the MCP App UI.

    Args:
        request: The tool call request.
        auth: The authentication handler.

    Returns:
        The tool call result including structured content for the UI.
    """
    logger.debug(
        "MCP Apps tool call: %s/%s with args: %s",
        request.server_name,
        request.tool_name,
        request.arguments,
    )

    server_config = get_mcp_server_config(request.server_name)

    try:
        async with streamablehttp_client(
            url=server_config["url"],
            headers=server_config.get("headers", {}),
            timeout=server_config.get("timeout", 30),
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                result = await session.call_tool(
                    request.tool_name,
                    arguments=request.arguments,
                )

                content_list = []
                for item in result.content:
                    match item.type:
                        case "text":
                            content_list.append({"type": "text", "text": item.text})
                        case "image":
                            content_list.append({
                                "type": "image",
                                "data": item.data,
                                "mimeType": item.mimeType,
                            })
                        case "audio":
                            content_list.append({
                                "type": "audio",
                                "data": item.data,
                                "mimeType": item.mimeType,
                            })
                        case _:
                            logger.warning(
                                "Unsupported content block type '%s' from tool '%s'",
                                getattr(item, "type", "unknown"),
                                request.tool_name,
                            )

                return MCPAppToolCallResponse(
                    content=content_list,
                    structured_content=getattr(result, "structuredContent", None),
                    is_error=getattr(result, "isError", False),
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error calling MCP tool: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to call tool: {e}",
        )

