"""Functions/Tools definition."""

import logging

from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools.base import BaseTool
from pydantic import ValidationError

from ols.src.tools.oc_cli import oc_adm_top, oc_describe, oc_get, oc_logs, oc_status

logger = logging.getLogger(__name__)

oc_tools = {
    "oc_get": oc_get,
    "oc_describe": oc_describe,
    "oc_logs": oc_logs,
    "oc_adm_top": oc_adm_top,
    "oc_status": oc_status,
}


def check_tool_description_length(tool: BaseTool) -> None:
    """Check if the description of a tool is too long."""
    if len(tool.description) >= 1024:
        raise ValueError(f"Tool {tool.name} description too long")


# NOTE: Limit for a tool description is 1024 characters
for tool in oc_tools.values():
    check_tool_description_length(tool)


def execute_oc_tool_calls(
    tools_map: dict,
    tool_calls: list[ToolCall],
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
            logger.error(tool_output)
        else:
            try:
                # inject token into tool args and immediately remove it
                # to avoid leaking
                tool_args["token"] = token
                tool_output = tool.invoke(tool_args)
                del tool_args["token"]
            except ValidationError:
                tool_output = (
                    f"Error executing {tool_name}: tool arguments are in wrong format"
                )
                # don't log as exception because it contains traceback
                # with sensitive information
                logger.error(tool_output)
            except Exception as e:
                tool_output = f"Error executing {tool_name}: {e}"
                logger.exception(tool_output)
            finally:
                # remove token from tool args if it was not removed
                # in the try block
                if "token" in tool_args:
                    del tool_args["token"]

        logger.debug(
            "Tool: %s | Args: %s | Output: %s", tool_name, tool_args, tool_output
        )

        tool_messages.append(ToolMessage(tool_output, tool_call_id=tool_call.get("id")))

    return tool_messages
