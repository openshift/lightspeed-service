"""Functions/Tools definition."""

import logging

from langchain_core.messages import ToolMessage
from langchain_core.tools.structured import StructuredTool

logger = logging.getLogger(__name__)


def get_tool_by_name(
    tool_name: str, all_mcp_tools: list[StructuredTool]
) -> StructuredTool:
    """Get a tool by its name from the MCP client."""
    tool = [tool for tool in all_mcp_tools if tool.name == tool_name]
    if len(tool) == 0:
        raise ValueError(f"Tool '{tool_name}' not found.")
    if len(tool) > 1:
        # TODO: LCORE-94
        raise ValueError(f"Multiple tools found with name '{tool_name}'.")
    return tool[0]


async def execute_tool_call(
    tool_name: str, tool_args: dict, all_mcp_tools: list[StructuredTool]
) -> tuple[str, str]:
    """Execute a tool call and return the output and status."""
    try:
        tool = get_tool_by_name(tool_name, all_mcp_tools)
        tool_output = await tool.arun(tool_args)  # type: ignore [attr-defined]
        status = "success"
        logger.debug(
            "Tool: %s | Args: %s | Output: %s", tool_name, tool_args, tool_output
        )
    except Exception as e:
        # catching generic exception here - if it contains something it
        # shouldn't (eg. token in openshift tools), it is responsibility
        # of the mcp/openshift tools to ensure nothing is leaked
        tool_output = f"Error executing tool '{tool_name}': {e}"
        status = "error"
        logger.exception(tool_output)
    return status, tool_output


async def execute_tool_calls(
    tool_calls: list[dict],
    all_mcp_tools: list[StructuredTool],
) -> tuple[list[ToolMessage], list[dict]]:
    """Execute tool calls and return ToolMessages and execution details."""
    tool_messages = []

    for tool_call in tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id")
        try:
            status, tool_output = await execute_tool_call(
                tool_name, tool_args, all_mcp_tools
            )
        except Exception as e:
            tool_output = (
                f"Error executing tool '{tool_name}' with args {tool_args}: {e}"
            )
            status = "error"
            logger.exception(tool_output)
        tool_messages.append(
            ToolMessage(content=tool_output, status=status, tool_call_id=tool_id)
        )

    return tool_messages


def filter_read_only_tools(
    all_mcp_tools: list[StructuredTool],
) -> list[StructuredTool]:
    """Filter out read-only tools from the list of all MCP tools."""
    read_only_tools = []
    for tool in all_mcp_tools:
        if tool.metadata is None:
            # TBD should we skip or add a tool with no metadata?
            logger.warning("Tool %s has no metadata", tool.name)
            read_only_tools.append(tool)
        elif tool.metadata.get("readOnlyHint", False):
            read_only_tools.append(tool)
    logger.debug(
        "Filtered read-only tools: %s", [tool.name for tool in read_only_tools]
    )
    return read_only_tools
