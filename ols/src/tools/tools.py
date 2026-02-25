"""Functions/Tools definition."""

import asyncio
import json
import logging
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools.structured import StructuredTool

from ols.constants import DEFAULT_MAX_TOKENS_PER_TOOL_OUTPUT
from ols.utils.token_handler import TokenHandler

logger = logging.getLogger(__name__)


SENSITIVE_KEYWORDS = ["secret"]


def _extract_text_from_tool_output(output: Any) -> str:
    """Extract plain text from tool output.

    Handle both old-style string output and new-style content block
    list output from langchain-mcp-adapters>=0.2.0 which returns
    LC standard content blocks like [{'type': 'text', 'text': '...'}].

    Args:
        output: Tool output, either a string or list of content blocks.

    Returns:
        Plain text string extracted from the output.
    """
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts = []
        for block in output:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(output)


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
    tool_name: str,
    tool_args: dict,
    all_mcp_tools: list[StructuredTool],
    max_tokens: int = DEFAULT_MAX_TOKENS_PER_TOOL_OUTPUT,
) -> tuple[str, str, bool, dict | None]:
    """Execute a tool call and return output, status, truncation flag, and structured content.

    Args:
        tool_name: Name of the tool to execute
        tool_args: Arguments to pass to the tool
        all_mcp_tools: List of available MCP tools
        max_tokens: Maximum tokens allowed for tool output

    Returns:
        Tuple of (status, tool_output, was_truncated, structured_content).
    """
    was_truncated = False
    structured_content: dict | None = None
    try:
        tool = get_tool_by_name(tool_name, all_mcp_tools)
        tool_output_raw, artifact = await tool.coroutine(**_jsonify(tool_args))  # type: ignore[misc]
        tool_output = _extract_text_from_tool_output(tool_output_raw)
        if isinstance(artifact, dict):
            structured_content = artifact.get("structured_content")

        token_handler = TokenHandler()
        tool_output, was_truncated = token_handler.truncate_tool_output(
            tool_output, max_tokens
        )

        status = "success"
        logger.debug(
            "Tool: %s | Args: %s | Output: %s | Truncated: %s"
            " | Has structured_content: %s",
            tool_name,
            tool_args,
            tool_output[:500] if len(tool_output) > 500 else tool_output,
            was_truncated,
            structured_content is not None,
        )
    except Exception as e:
        tool_output = f"Error executing tool '{tool_name}': {e}"
        status = "error"
        logger.exception(tool_output)
    return status, tool_output, was_truncated, structured_content


async def _execute_single_tool_call(
    tool_call: dict,
    all_mcp_tools: list[StructuredTool],
    max_tokens: int = DEFAULT_MAX_TOKENS_PER_TOOL_OUTPUT,
) -> ToolMessage:
    """Execute a single tool call and return a ToolMessage.

    Args:
        tool_call: Tool call dict with name, args, and id.
        all_mcp_tools: List of available MCP tools.
        max_tokens: Maximum tokens allowed for tool output.

    Returns:
        ToolMessage with result. The additional_kwargs contains:
        - "truncated": bool indicating if output was truncated
        - "structured_content": dict with structured data (if available)
    """
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args", {})
    tool_id = tool_call.get("id")
    was_truncated = False
    structured_content: dict | None = None

    if tool_name is None:
        tool_output = "Error: Tool name is missing from tool call"
        status = "error"
        logger.error("Tool call missing name: %s", tool_call)
    else:
        try:
            raise_for_sensitive_tool_args(tool_args)
            status, tool_output, was_truncated, structured_content = (
                await execute_tool_call(tool_name, tool_args, all_mcp_tools, max_tokens)
            )
        except Exception as e:
            tool_output = (
                f"Error executing tool '{tool_name}' with args {tool_args}: {e}"
            )
            status = "error"
            logger.exception(tool_output)

    additional_kwargs: dict = {"truncated": was_truncated}
    if structured_content is not None:
        additional_kwargs["structured_content"] = structured_content

    return ToolMessage(
        content=tool_output,
        status=status,
        tool_call_id=tool_id,
        additional_kwargs=additional_kwargs,
    )


async def execute_tool_calls(
    tool_calls: list[dict],
    all_mcp_tools: list[StructuredTool],
    max_tokens_per_output: int = DEFAULT_MAX_TOKENS_PER_TOOL_OUTPUT,
) -> list[ToolMessage]:
    """Execute tool calls in parallel and return ToolMessages.

    Args:
        tool_calls: List of tool call dicts
        all_mcp_tools: List of available MCP tools
        max_tokens_per_output: Maximum tokens allowed per tool output

    Returns:
        List of ToolMessages. Each message's additional_kwargs["truncated"]
        indicates if that tool's output was truncated.
    """
    if not tool_calls:
        return []

    tasks = [
        _execute_single_tool_call(tool_call, all_mcp_tools, max_tokens_per_output)
        for tool_call in tool_calls
    ]

    tool_messages = await asyncio.gather(*tasks)

    return tool_messages


def _jsonify(args: dict) -> dict:
    """Convert to JSON."""
    res = {}
    for key, value in args.items():
        if isinstance(value, str) and _maybe_json(value):
            # If a value looks like json
            try:
                # convert to json
                res[key] = json.loads(value)
            except json.JSONDecodeError:
                # conversion fails, use a string
                res[key] = value
        else:
            res[key] = value
    return res


def _maybe_json(value: str) -> bool:
    """Check if a string looks like JSON."""
    stripped = value.strip()
    return stripped.startswith(("[", "{"))
