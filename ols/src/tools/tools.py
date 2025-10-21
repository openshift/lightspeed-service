"""Functions/Tools definition."""

import asyncio
import json
import logging

from langchain_core.messages import ToolMessage
from langchain_core.tools.structured import StructuredTool
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


SENSITIVE_KEYWORDS = ["secret"]


def raise_for_sensitive_tool_args(tool_args: dict) -> None:
    """Check tool arguments for sensitive content and raise an exception if found."""
    for key, value in tool_args.items():
        if any(keyword in str(value).lower() for keyword in SENSITIVE_KEYWORDS):
            raise ValueError(
                f"Sensitive keyword in tool arguments {key}={value} are not allowed."
            )


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


async def _execute_single_tool_call(
    tool_call: dict, all_mcp_tools: list[StructuredTool]
) -> ToolMessage:
    """Execute a single tool call and return a ToolMessage."""
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args", {})
    tool_id = tool_call.get("id")

    if tool_name is None:
        tool_output = "Error: Tool name is missing from tool call"
        status = "error"
        logger.error("Tool call missing name: %s", tool_call)
    else:
        try:
            # generate_ui args contain live data about e.g. pods containing sensitive words like secret etc.
            if tool_name != "generate_ui":
                raise_for_sensitive_tool_args(tool_args)
            status, tool_output = await execute_tool_call(
                tool_name, tool_args, all_mcp_tools
            )
        except Exception as e:
            tool_output = (
                f"Error executing tool '{tool_name}' with args {tool_args}: {e}"
            )
            status = "error"
            logger.exception(tool_output)

    return ToolMessage(
        content=tool_output,
        status=status,
        tool_call_id=tool_id,
        name=tool_call.get("name"),
    )


async def execute_tool_calls(
    tool_calls: list[dict],
    all_mcp_tools: list[StructuredTool],
    messages: ChatPromptTemplate,
) -> list[ToolMessage]:
    """Execute tool calls in parallel and return ToolMessages."""
    if not tool_calls:
        return []

    # Create tasks for parallel execution except generate_ui.
    # generate_ui task from next_gen_ui MCP has to collect data from other MCP first
    tasks = [
        _execute_single_tool_call(tool_call, all_mcp_tools)
        for tool_call in tool_calls
        if not tool_call["name"] == "generate_ui"
    ]

    # Execute all tool calls in parallel
    tool_messages = await asyncio.gather(*tasks)

    # Execute next_gen_ui tool with all data from previous tools
    # TODO: Following code should refactored to use "memory/context" where
    # each MCP can access "short term memory (context)" and act appropriately
    # MCP has no support for this ATM

    generate_ui_task = next(
        (tool_call for tool_call in tool_calls if tool_call["name"] == "generate_ui"),
        None,
    )
    if generate_ui_task:
        # get all tool responses and pass to NGUI arg
        generate_ui_task["args"]["structured_data"] = []
        if len(tool_messages) > 0:
            ngui_input_data = [
                {
                    "id": tm.tool_call_id,
                    "data": tm.content,
                    "type": tm.name,
                }
                for tm in tool_messages
            ]
            generate_ui_task["args"]["structured_data"].extend(ngui_input_data)
        else:
            ngui_input_data = [
                {
                    "id": tm.tool_call_id,
                    "data": str(tm.content),
                    "type": tm.name,
                }
                for tm in messages.messages
                if isinstance(tm, ToolMessage)
            ]
            generate_ui_task["args"]["structured_data"].extend(ngui_input_data)

        logger.info(
            "Executing generate_ui tool_call. input_data.length=%s",
            len(generate_ui_task["args"]["structured_data"]),
        )

        generate_ui_result = await _execute_single_tool_call(
            generate_ui_task, all_mcp_tools
        )

        # Change Content and Artifact
        # Putting into artifact is a standard way how to send data to client and NOT TO LLM
        # NGUI support that natively but Current OLS MCP client has no support for MCP structured_content.
        # So NGUI has structured_output disabled and it's needed to do it manually
        if generate_ui_result.status == "success":
            generate_ui_result.artifact = generate_ui_result.content
            # Get summary. https://redhat-ux.github.io/next-gen-ui-agent/guide/ai_apps_binding/mcp-library/#generate_ui
            generate_ui_content: dict = json.loads(generate_ui_result.content)
            # This tells LLM not to repeat again what is displayed on dashboard.
            generate_ui_result.content = generate_ui_content.get("summary")

        tool_messages.append(generate_ui_result)

    return tool_messages
