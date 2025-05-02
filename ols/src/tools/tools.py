"""Functions/Tools definition."""

import logging
from typing import Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools.base import BaseTool
from pydantic import ValidationError

from ols.src.tools.oc_cli import (
    oc_adm_top,
    oc_describe,
    oc_get,
    oc_logs,
    oc_status,
    show_pods_resource_usage,
)

logger = logging.getLogger(__name__)


oc_tools = {
    "oc_get": oc_get,
    "oc_describe": oc_describe,
    "oc_logs": oc_logs,
    "oc_adm_top": oc_adm_top,
    "oc_status": oc_status,
    "show_pods_resource_usage": show_pods_resource_usage,
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

    logger.info("User token provided; adding 'oc' tools")
    return oc_tools


def check_tool_description_length(tool: BaseTool) -> None:
    """Check if the description of a tool is too long."""
    if len(tool.description) >= 1024:
        raise ValueError(f"Tool {tool.name} description too long")


# NOTE: Limit for a tool description is 1024 characters
for tool in oc_tools.values():
    check_tool_description_length(tool)


def execute_oc_tool_calls(
    tools_map: dict,
    tool_calls: list[dict],
    token: str,
) -> tuple[list[ToolMessage], list[dict]]:
    """Execute tool calls and return ToolMessages and execution details."""
    tool_messages = []

    for tool_call in tool_calls:
        tool_name = tool_call.get("name", "").lower()
        tool_args = tool_call.get("args", {})
        tool = tools_map.get(tool_name)

        if not tool:
            tool_output = f"Error: Tool '{tool_name}' not found."
            status = "error"
            logger.error(tool_output)
        else:
            try:
                # create a new dict with the tool args and the token
                status, tool_output = tool.invoke({**tool_args, "token": token})
                logger.debug(
                    "Tool: %s | Args: %s | Output: %s",
                    tool_name,
                    tool_args,
                    tool_output,
                )
            except ValidationError as e:
                tool_output = f'Tool arguments are in wrong format: {str(e).replace(token, "<redacted>")}'  # noqa E501
                status = "error"
                # don't log as exception because it contains traceback
                # with sensitive information
                logger.error(tool_output)
            except Exception as e:
                tool_output = f"Error executing tool '{tool_name}': {str(e).replace(token, '<redacted>')}"  # noqa E501
                status = "error"
                logger.error(tool_output)

        tool_messages.append(
            ToolMessage(
                content=tool_output, status=status, tool_call_id=tool_call.get("id")
            )
        )

    return tool_messages
