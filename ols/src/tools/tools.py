"""Functions/Tools definition."""

import asyncio
import json
import logging

from langchain_core.messages import ToolMessage, BaseMessage
from langchain_core.tools.structured import StructuredTool

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
            if not tool_name.startswith("generate_ui"):
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
    messages: list[BaseMessage],
) -> list[ToolMessage]:
    """Execute tool calls in parallel and return ToolMessages."""
    if not tool_calls:
        return []

    # Create tasks for parallel execution
    tasks = [
        _execute_single_tool_call(tool_call, all_mcp_tools)
        for tool_call in tool_calls
        if not tool_call["name"] == "generate_ui_multiple_components"
    ]

    # Execute all tool calls in parallel
    tool_messages = await asyncio.gather(*tasks)

    generate_ui_results = await execute_tool_generate_ui_calls(
        tool_calls, all_mcp_tools, messages
    )

    tool_messages.extend(generate_ui_results)

    return tool_messages


async def execute_tool_generate_ui_calls(
    tool_calls: list[dict],
    all_mcp_tools: list[StructuredTool],
    messages: list[BaseMessage],
) -> ToolMessage | None:
    """
    Function to execute Next Gen UI tool calls.
    """
    # Find NGUI tool calls:
    # https://redhat-ux.github.io/next-gen-ui-agent/guide/ai_apps_binding/mcp-library/#available-mcp-tools
    generate_ui_tasks = [
        tool_call
        for tool_call in tool_calls
        if tool_call["name"] == "generate_ui_multiple_components"
        or tool_call["name"] == "generate_ui_component"
    ]
    generate_ui_results: list[ToolMessage] = []
    for generate_ui_task in generate_ui_tasks:
        if generate_ui_task["name"] == "generate_ui_component":
            # generate_ui_component tool call expect data in plain string arguments.
            # Better LLMs "can" provide that
            tool_result = await _execute_single_tool_call(
                generate_ui_task, all_mcp_tools
            )
        elif generate_ui_task["name"] == "generate_ui_multiple_components":
            # generate_ui_multiple_components expect list of components.
            # To optimize data collection, get all tool messages from current agent's turn
            # TODO: Consider this for "generate_ui_component" tool call as well
            ngui_input_data = [
                {
                    "id": tm.tool_call_id,
                    "data": tm.content,
                    "type": tm.name,
                }
                for tm in messages
                if isinstance(tm, ToolMessage)
            ]
            generate_ui_task["args"]["structured_data"] = ngui_input_data

            tool_result = await _execute_single_tool_call(
                generate_ui_task, all_mcp_tools
            )

        generate_ui_results.append(tool_result)

    for generate_ui_result in generate_ui_results:
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

    return generate_ui_results
