"""Functions/Tools definition."""

import logging
from typing import Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools.base import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from ols.src.tools.oc_cli import (
    oc_adm_top,
    oc_describe,
    oc_get,
    oc_logs,
    oc_status,
    show_pods,
    token_works_for_oc,
)

logger = logging.getLogger(__name__)


oc_tools = {
    "oc_get": oc_get,
    "oc_describe": oc_describe,
    "oc_logs": oc_logs,
    "oc_adm_top": oc_adm_top,
    "oc_status": oc_status,
    "show_pods": show_pods,
}


def get_available_tools(
    introspection_enabled: bool, user_token: Optional[str] = None
) -> dict[str, BaseTool]:
    """Get available tools based on introspection and user token."""
    if not introspection_enabled:
        logger.info("Introspection disabled; no tools available")
        return {}

    if user_token is None or (isinstance(user_token, str) and user_token == ""):
        logger.warning("No user token provided; no oc tools available")
        return {}

    if token_works_for_oc(user_token):
        logger.info("Authenticated to 'oc' CLI; adding 'oc' tools")
        return oc_tools

    logger.error("User token not working for 'oc' CLI; no tools available")
    return {}


def check_tool_description_length(tool: BaseTool) -> None:
    """Check if the description of a tool is too long."""
    if len(tool.description) >= 1024:
        raise ValueError(f"Tool {tool.name} description too long")


# NOTE: Limit for a tool description is 1024 characters
for tool in oc_tools.values():
    check_tool_description_length(tool)


def get_tool_by_name(tool_name: str, mcp_client: MultiServerMCPClient):
    """Get a tool by its name from the MCP client."""
    tool = [tool for tool in mcp_client.get_tools() if tool.name == tool_name]
    if len(tool) == 0:
        raise ValueError(f"Tool '{tool_name}' not found.")
    if len(tool) > 1:
        raise ValueError(f"Multiple tools found with name '{tool_name}'.")
    return tool[0]


async def execute_tool_call(tool_call: dict, mcp_client: MultiServerMCPClient):
    """Execute a tool call and return the output and status."""
    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("args", {})
    try:
        tool = get_tool_by_name(tool_name, mcp_client)
        tool_output = await tool.arun(tool_args)
        status = "success"
        logger.debug(
            "Tool: %s | Args: %s | Output: %s", tool_name, tool_args, tool_output
        )
    except Exception as e:
        # catching generic exception here - if it contains something it
        # shouldn't (eg. token in openshift tools), it is responsibility
        # of the mcp/openshift tools to ensure nothing is leaked
        tool_output = f"Error executing tool '{tool_call['name']}': {e}"
        status = "error"
        logger.exception(tool_output)
    return tool_output, status


async def execute_tool_calls(
    mcp_client: MultiServerMCPClient,
    tool_calls: list[dict],
) -> tuple[list[ToolMessage], list[dict]]:
    """Execute tool calls and return ToolMessages and execution details."""
    tool_messages = []

    for tool_call in tool_calls:
        tool_output, status = await execute_tool_call(tool_call, mcp_client)

        tool_messages.append(
            ToolMessage(
                content=tool_output, status=status, tool_call_id=tool_call.get("id")
            )
        )

    return tool_messages
