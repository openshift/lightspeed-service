"""Handlers for MCP server information endpoints."""

import logging

from fastapi import APIRouter, Depends

from ols import config
from ols.app.models.models import MCPHeadersResponse, MCPServerHeaderInfo
from ols.src.auth.auth import get_auth_dependency
from ols.utils.mcp_utils import get_servers_requiring_client_headers

logger = logging.getLogger(__name__)

router = APIRouter(tags=["mcp"])
auth_dependency = get_auth_dependency(config.ols_config, virtual_path="/ols-access")


@router.get(
    "/mcp/client-auth-headers",
    response_model=MCPHeadersResponse,
    summary="Get MCP servers requiring client authorization headers",
    description=(
        "Returns information about which MCP servers require client-provided "
        "authorization headers. Clients should include these headers in the "
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
