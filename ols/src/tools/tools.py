"""Functions/Tools definition."""

import asyncio
import logging
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools.structured import StructuredTool

from ols.utils.token_handler import TokenHandler

logger = logging.getLogger(__name__)


MAX_TOOL_CALL_RETRIES = 2
RETRY_BACKOFF_SECONDS = 0.2
DO_NOT_RETRY_REMINDER = "Do not retry this exact tool call."

_TRUNCATION_WARNING = (
    "\n\n[OUTPUT TRUNCATED - The tool returned more data than can be "
    "processed. Please ask a more specific question to get complete results.]"
)
_TRUNCATION_WARNING_TOKENS = TokenHandler._get_token_count(
    TokenHandler().text_to_tokens(_TRUNCATION_WARNING)
)


def _is_retryable_tool_error(error: Exception) -> bool:
    """Return true if a tool execution error is likely transient."""
    if isinstance(error, (TimeoutError, asyncio.TimeoutError, ConnectionError)):
        return True
    error_text = str(error).lower()
    return any(
        token in error_text
        for token in (
            "timeout",
            "temporarily unavailable",
            "temporary failure",
            "connection reset",
            "connection closed",
        )
    )


_CHARS_PER_TOKEN_ESTIMATE = 4


def _extract_text_from_tool_output(
    output: Any, tools_token_budget: int
) -> tuple[str, bool]:
    """Extract plain text from tool output with a character-level size guard.

    Handle both old-style string output and new-style content block
    list output from langchain-mcp-adapters>=0.2.0 which returns
    LC standard content blocks like [{'type': 'text', 'text': '...'}].

    Neither LangChain nor the MCP SDK provide a mechanism to limit tool
    response size at the transport layer, so a cheap character-level limit
    (tools_token_budget * 4) is applied here before any tokenization to
    avoid the CPU cost of tokenizing arbitrarily large tool responses.
    Strings are cut at the last newline boundary; lists are truncated by
    dropping trailing blocks that would exceed the limit.

    Args:
        output: Tool output, either a string or list of content blocks.
        tools_token_budget: Remaining token budget for tool outputs; used
            to derive a cheap character limit so we never tokenize an
            arbitrarily large string.

    Returns:
        Tuple of (extracted text, was_truncated).
    """
    max_chars = tools_token_budget * _CHARS_PER_TOKEN_ESTIMATE

    if not isinstance(output, list):
        output = str(output)
        if len(output) <= max_chars:
            return output, False
        cut = output[:max_chars].rfind("\n")
        text = output[:cut].rstrip("\r") if cut > 0 else output[:max_chars]
        logger.warning(
            "Tool output pre-truncated from %d to %d chars (limit %d)",
            len(output),
            len(text),
            max_chars,
        )
        return text + _TRUNCATION_WARNING, True

    parts: list[str] = []
    total = 0
    for block in output:
        chunk = (
            block["text"] if isinstance(block, dict) and "text" in block else str(block)
        )
        if total + len(chunk) > max_chars:
            logger.warning(
                "Tool output pre-truncated at block %d of %d (limit %d chars)",
                len(parts),
                len(output),
                max_chars,
            )
            return "\n".join(parts) + _TRUNCATION_WARNING, True
        parts.append(chunk)
        total += len(chunk)

    return "\n".join(parts), False


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
    tools_token_budget: int,
) -> tuple[str, bool, dict | None]:
    """Execute a tool call and return output, truncation flag, and structured content.

    Args:
        tool_name: Name of the tool to execute
        tool_args: Arguments to pass to the tool
        all_mcp_tools: List of available MCP tools
        tools_token_budget: Remaining token budget for tool outputs

    Returns:
        Tuple of (tool_output, was_truncated, structured_content).
    """
    structured_content: dict | None = None
    tool = get_tool_by_name(tool_name, all_mcp_tools)
    tool_output_raw, artifact = await tool.coroutine(**tool_args)  # type: ignore[misc]
    tool_output, was_truncated = _extract_text_from_tool_output(
        tool_output_raw, tools_token_budget
    )
    if isinstance(artifact, dict):
        structured_content = artifact.get("structured_content")

    logger.debug(
        "Tool: %s | Args: %s | Output: %s | Truncated: %s"
        " | Has structured_content: %s",
        tool_name,
        tool_args,
        tool_output[:200] if len(tool_output) > 200 else tool_output,
        was_truncated,
        structured_content is not None,
    )
    return tool_output, was_truncated, structured_content


async def _execute_single_tool_call(
    tool_call: dict,
    all_mcp_tools: list[StructuredTool],
    tools_token_budget: int,
) -> ToolMessage:
    """Execute a single tool call and return a ToolMessage.

    Args:
        tool_call: Tool call dict with name, args, and id.
        all_mcp_tools: List of available MCP tools.
        tools_token_budget: Remaining token budget for tool outputs.

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
        tool_output = (
            f"Error: Tool name is missing from tool call. {DO_NOT_RETRY_REMINDER}"
        )
        status = "error"
        logger.error("Tool call missing name: %s", tool_call)
    else:
        attempts = MAX_TOOL_CALL_RETRIES + 1
        last_error_text = "unknown error"
        for attempt in range(attempts):
            try:
                tool_output, was_truncated, structured_content = (
                    await execute_tool_call(
                        tool_name, tool_args, all_mcp_tools, tools_token_budget
                    )
                )
                status = "success"
                break
            except Exception as error:
                last_error_text = str(error)
                if attempt < MAX_TOOL_CALL_RETRIES and _is_retryable_tool_error(error):
                    logger.warning(
                        "Retrying tool '%s' after transient error on attempt %d/%d: %s",
                        tool_name,
                        attempt + 1,
                        attempts,
                        error,
                    )
                    await asyncio.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))
                    continue
                tool_output = (
                    f"Tool '{tool_name}' execution failed after {attempt + 1} "
                    f"attempt(s): {last_error_text}. {DO_NOT_RETRY_REMINDER}"
                )
                status = "error"
                logger.exception(tool_output)
                break

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
    tools_token_budget: int,
) -> list[ToolMessage]:
    """Execute tool calls in parallel and return ToolMessages.

    Args:
        tool_calls: List of tool call dicts
        all_mcp_tools: List of available MCP tools
        tools_token_budget: Remaining token budget for tool outputs

    Returns:
        List of ToolMessages. Each message's additional_kwargs["truncated"]
        indicates if that tool's output was truncated.
    """
    if not tool_calls:
        return []

    tasks = [
        _execute_single_tool_call(tool_call, all_mcp_tools, tools_token_budget)
        for tool_call in tool_calls
    ]

    tool_messages = await asyncio.gather(*tasks)

    return tool_messages


def enforce_tool_token_budget(
    tool_messages: list[ToolMessage],
    remaining_budget: int,
) -> list[ToolMessage]:
    """Ensure combined tool outputs fit within the remaining token budget.

    Uses a three-tier strategy to avoid unnecessary tokenization:
    1. Cheap character-based estimate — skip tokenization if clearly under budget.
    2. Precise tokenization — only if the estimate suggests overflow.
    3. Truncation — if the longest message dominates (>= 2x excess), only it
       is shrunk; otherwise all messages are scaled proportionally.

    Args:
        tool_messages: Tool result messages to enforce budget on.
        remaining_budget: Remaining token budget for tool outputs.

    Returns:
        The same list of ToolMessages, with oversized ones truncated in place.
    """
    if not tool_messages:
        return tool_messages

    # Tier 1: cheap char-based estimate (~4 chars/token). The 0.9 factor
    # compensates for the approximation; if we're clearly under budget,
    # skip the expensive tokenization entirely.
    estimated_tokens = sum(
        len(str(msg.content)) // _CHARS_PER_TOKEN_ESTIMATE for msg in tool_messages
    )
    if estimated_tokens <= int(remaining_budget * 0.9):
        return tool_messages

    # Tier 2: precise tokenization. The char estimate was ambiguous,
    # so tokenize each message to get exact counts.
    token_handler = TokenHandler()
    token_lists = [
        token_handler.text_to_tokens(str(msg.content)) for msg in tool_messages
    ]
    token_counts = [TokenHandler._get_token_count(t) for t in token_lists]
    total = sum(token_counts)
    if total <= remaining_budget:
        return tool_messages

    logger.warning(
        "Tool outputs (%d tokens) exceed remaining budget (%d), truncating",
        total,
        remaining_budget,
    )

    excess = total - remaining_budget
    longest_idx = max(range(len(token_counts)), key=lambda i: token_counts[i])

    # Tier 3: if the longest message alone can absorb the excess while
    # retaining at least half its content, shrink only that one.
    # Otherwise scale all messages proportionally to fit the budget.
    if token_counts[longest_idx] // 2 >= excess:
        targets = [longest_idx]
        limits = [token_counts[longest_idx] - excess]
    else:
        ratio = remaining_budget / total
        targets = list(range(len(token_counts)))
        limits = [max(1, int(token_counts[i] * ratio)) for i in targets]

    # Truncate targeted messages using pre-computed token lists (no
    # re-tokenization). Cut at the last newline to avoid mid-line splits.
    for idx, limit in zip(targets, limits):
        if token_counts[idx] <= limit:
            continue
        raw = token_handler.tokens_to_text(
            token_lists[idx][: max(0, limit - _TRUNCATION_WARNING_TOKENS)]
        )
        cut = raw.rfind("\n")
        truncated_text = (
            raw[:cut].rstrip("\r") if cut > 0 else raw
        ) + _TRUNCATION_WARNING
        msg = tool_messages[idx]
        tool_messages[idx] = ToolMessage(
            content=truncated_text,
            status=msg.status,
            tool_call_id=msg.tool_call_id,
            additional_kwargs={**msg.additional_kwargs, "truncated": True},
        )

    return tool_messages
