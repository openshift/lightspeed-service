"""Handlers for MCP server information endpoints."""

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ols import config
from ols.src.auth.auth import get_auth_dependency
from ols.utils.mcp_headers import get_servers_requiring_client_headers

logger = logging.getLogger(__name__)

router = APIRouter(tags=["mcp"])
auth_dependency = get_auth_dependency(config.ols_config, virtual_path="/ols-access")


class MCPServerHeaderInfo(BaseModel):
    """Information about headers required for an MCP server."""

    server_name: str = Field(..., description="Name of the MCP server")
    required_headers: list[str] = Field(
        ..., description="List of header names that client must provide"
    )


class MCPHeadersResponse(BaseModel):
    """Response model listing servers that require client headers."""

    servers: list[MCPServerHeaderInfo] = Field(
        ..., description="List of servers requiring client-provided headers"
    )


@router.get(
    "/mcp-requirements",
    response_model=MCPHeadersResponse,
    summary="Get MCP servers requiring client headers",
    description=(
        "Returns information about which MCP servers require client-provided "
        "authentication headers. Clients should include these headers in the "
        "MCP-Headers HTTP header when making queries."
    ),
    responses={
        200: {
            "description": "List of servers requiring client headers",
            "model": MCPHeadersResponse,
        },
    },
)
async def get_mcp_header_info(
    user_id: str = Depends(auth_dependency),
) -> MCPHeadersResponse:
    """Get information about MCP servers requiring client headers.

    Args:
        user_id: Authenticated user ID from dependency

    Returns:
        MCPHeadersResponse with list of servers and their required headers
    """
    servers_dict = get_servers_requiring_client_headers(config.mcp_servers)

    servers_info = [
        MCPServerHeaderInfo(server_name=name, required_headers=headers)
        for name, headers in servers_dict.items()
    ]

    logger.info(
        "Returning MCP header info for %d servers requiring client headers",
        len(servers_info),
    )

    return MCPHeadersResponse(servers=servers_info)
