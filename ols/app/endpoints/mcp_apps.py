"""MCP Apps proxy endpoints for UI resource fetching and tool calling.

These endpoints let the console (MCP host UI) fetch ui:// resources and
proxy tool calls to MCP servers on behalf of app iframes.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

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
from ols.utils.mcp_utils import resolve_server_headers

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp-apps", tags=["mcp-apps"])
auth_dependency = get_auth_dependency(config.ols_config, virtual_path="/ols-access")


def _get_server_config(server_name: str) -> dict[str, Any]:
    """Look up an MCP server by name and return its connection parameters.

    Args:
        server_name: Name of the MCP server as defined in olsconfig.yaml.

    Returns:
        Dict with url, timeout, and resolved_headers for the server.

    Raises:
        HTTPException 404: If no MCP servers are configured or the name is unknown.
    """
    if not config.mcp_servers or not config.mcp_servers.servers:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No MCP servers configured",
        )

    server = config.mcp_servers_dict.get(server_name)
    if server is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"MCP server '{server_name}' not found",
        )

    headers = resolve_server_headers(server, user_token=None, client_headers=None)

    return {
        "url": server.url,
        "timeout": server.timeout or 30,
        "headers": headers or {},
    }


resource_responses: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "Resource fetched successfully",
        "model": MCPAppResourceResponse,
    },
    400: {"description": "Invalid resource URI", "model": ErrorResponse},
    401: {
        "description": "Missing or invalid credentials",
        "model": UnauthorizedResponse,
    },
    403: {"description": "Permission denied", "model": ForbiddenResponse},
    404: {"description": "Server or resource not found", "model": ErrorResponse},
    500: {"description": "Internal server error", "model": ErrorResponse},
}


@router.post("/resources", responses=resource_responses)
async def get_mcp_app_resource(
    request: MCPAppResourceRequest,
    user_id: str = Depends(auth_dependency),
) -> MCPAppResourceResponse:
    """Fetch a ui:// resource from an MCP server.

    Args:
        request: Resource request with URI and server name.
        user_id: Authenticated user ID.

    Returns:
        The resource content (HTML/JS/CSS) with metadata.
    """
    if not request.resource_uri.startswith("ui://") or len(request.resource_uri) <= 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid resource URI: {request.resource_uri}. Expected ui://...",
        )

    server_config = _get_server_config(request.server_name)

    try:
        async with streamable_http_client(
            url=server_config["url"],
            headers=server_config["headers"],
            timeout=server_config["timeout"],
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
                is_text = hasattr(content, "text")

                return MCPAppResourceResponse(
                    uri=str(content.uri),
                    mime_type=getattr(content, "mimeType", None) or "text/html",
                    content=content.text if is_text else getattr(content, "blob", ""),
                    content_type="text" if is_text else "blob",
                    meta=getattr(content, "meta", None),
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to fetch MCP resource '%s': %s", request.resource_uri, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch resource: {e}",
        )


tool_call_responses: dict[int | str, dict[str, Any]] = {
    200: {"description": "Tool call successful", "model": MCPAppToolCallResponse},
    401: {
        "description": "Missing or invalid credentials",
        "model": UnauthorizedResponse,
    },
    403: {"description": "Permission denied", "model": ForbiddenResponse},
    404: {"description": "Server not found", "model": ErrorResponse},
    500: {"description": "Internal server error", "model": ErrorResponse},
}


@router.post("/tools/call", responses=tool_call_responses)
async def call_mcp_app_tool(
    request: MCPAppToolCallRequest,
    user_id: str = Depends(auth_dependency),
) -> MCPAppToolCallResponse:
    """Proxy a tool call from an app iframe to an MCP server.

    Args:
        request: Tool call request with server, tool name, and arguments.
        user_id: Authenticated user ID.

    Returns:
        Tool call result with content blocks and optional structured content.
    """
    server_config = _get_server_config(request.server_name)

    try:
        async with streamable_http_client(
            url=server_config["url"],
            headers=server_config["headers"],
            timeout=server_config["timeout"],
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    request.tool_name,
                    arguments=request.arguments,
                )

                content_list: list[dict[str, Any]] = []
                for item in result.content:
                    match item.type:
                        case "text":
                            content_list.append({"type": "text", "text": item.text})
                        case "image":
                            content_list.append(
                                {
                                    "type": "image",
                                    "data": item.data,
                                    "mimeType": item.mimeType,
                                }
                            )
                        case "audio":
                            content_list.append(
                                {
                                    "type": "audio",
                                    "data": item.data,
                                    "mimeType": item.mimeType,
                                }
                            )
                        case _:
                            logger.warning(
                                "Unsupported content type '%s' from tool '%s'",
                                item.type,
                                request.tool_name,
                            )

                return MCPAppToolCallResponse(
                    content=content_list,
                    structured_content=result.structuredContent,
                    is_error=result.isError,
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to call tool '%s' on server '%s': %s",
            request.tool_name,
            request.server_name,
            e,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to call tool: {e}",
        )
