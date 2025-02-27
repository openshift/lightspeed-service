"""Functions/Tools definition."""

import logging

from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools.base import BaseTool

from ols.src.tools.oc_cli import oc_adm_top, oc_describe, oc_get, oc_logs, oc_status

logger = logging.getLogger(__name__)

oc_tools = {
    "oc_get": oc_get,
    "oc_describe": oc_describe,
    "oc_logs": oc_logs,
    "oc_adm_top": oc_adm_top,
    "oc_status": oc_status,
}

# Default tools map
default_tools: dict = {}


def check_tool_description_length(tool: BaseTool) -> None:
    """Check if the description of a tool is too long."""
    if len(tool.description) >= 1024:
        raise ValueError(f"Tool {tool.name} description too long")


# NOTE: Limit for a tool description is 1024 characters
for tool in {**default_tools, **oc_tools}.values():
    check_tool_description_length(tool)


def execute_tool_calls(
    tools_map: dict, tool_calls: list[ToolCall]
) -> list[ToolMessage]:
    """Execute tool calls and return ToolMessages."""
    tool_messages = []
    for tool_call in tool_calls:
        tool_name = tool_call.get("name", "").lower()
        tool_args = tool_call.get("args", {})

        tool = tools_map.get(tool_name)
        if not tool:
            logger.error("Tool '%s' not found in tools map.", tool_name)
            tool_output = f"Error: Tool '{tool_name}' not found."
        else:
            try:
                tool_output = tool.invoke(tool_args)
            except Exception as e:
                logger.exception("Error executing tool '%s'", tool_name)
                tool_output = f"Error executing {tool_name}: {e!s}"

        logger.debug(
            "Tool call: %s, Args: %s, Output: %s", tool_name, tool_args, tool_output
        )
        tool_messages.append(ToolMessage(tool_output, tool_call_id=tool_call.get("id")))

    return tool_messages
