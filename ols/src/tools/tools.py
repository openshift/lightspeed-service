"""Functions/Tools definition."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypedDict
from uuid import uuid4

from aiostream import stream
from langchain_core.messages import ToolMessage
from langchain_core.tools.structured import StructuredTool

from ols import config
from ols.app.models.models import StreamChunkType
from ols.src.tools.approval import (
    get_approval_decision,
    need_validation,
    normalize_tool_annotation,
    register_pending_approval,
)
from ols.utils.token_handler import TokenHandler

if TYPE_CHECKING:
    from ols.src.tools.offloaded_content import OffloadManager

logger = logging.getLogger(__name__)


MAX_TOOL_CALL_RETRIES = 2
RETRY_BACKOFF_SECONDS = 0.2
RATE_LIMIT_RETRY_BACKOFF_SECONDS = 1.0
DO_NOT_RETRY_REMINDER = "Do not retry this exact tool call."

_TRUNCATION_WARNING = (
    "\n\n[OUTPUT TRUNCATED - The tool returned more data than can be "
    "processed. Please ask a more specific question to get complete results.]"
)
_TRUNCATION_WARNING_TOKENS = TokenHandler._get_token_count(
    TokenHandler().text_to_tokens(_TRUNCATION_WARNING)
)


class ApprovalRequiredPayload(TypedDict):
    """Payload for approval_required events."""

    approval_id: str
    tool_name: str
    tool_description: str
    tool_args: dict[str, object]
    tool_annotation: dict[str, object]


@dataclass(slots=True)
class ApprovalRequiredEvent:
    """Approval-required event emitted during tool execution."""

    data: ApprovalRequiredPayload
    event: Literal[StreamChunkType.APPROVAL_REQUIRED] = (
        StreamChunkType.APPROVAL_REQUIRED
    )


@dataclass(slots=True)
class ToolResultEvent:
    """Tool-result event emitted during tool execution."""

    data: ToolMessage
    event: Literal[StreamChunkType.TOOL_RESULT] = StreamChunkType.TOOL_RESULT


ToolExecutionEvent = ApprovalRequiredEvent | ToolResultEvent
ToolCallDefinition: TypeAlias = tuple[str, dict[str, object], StructuredTool]


class _ApprovalNotGrantedError(Exception):
    """Internal control-flow signal when approval is denied or times out."""


def _is_transient_tool_error(error: Exception) -> bool:
    """Return true if a tool execution error is likely transient."""
    if isinstance(error, (TimeoutError, asyncio.TimeoutError, ConnectionError)):
        return True
    if isinstance(error, OSError):
        return True
    error_text = str(error).lower()
    return any(
        token in error_text
        for token in ("timeout", "temporar", "connection reset", "connection closed")
    )


def _is_rate_limited_tool_error(error: Exception) -> bool:
    """Return true if a tool execution error indicates rate-limiting."""
    error_text = str(error).lower()
    return any(
        token in error_text for token in ("rate limit", "429", "too many requests")
    )


_CHARS_PER_TOKEN_ESTIMATE = 4


def _convert_tool_output_to_text(output: Any) -> str:
    """Convert tool output to plain text without applying size limits.

    Handle both old-style string output and new-style content block
    list output from langchain-mcp-adapters>=0.2.0 which returns
    LC standard content blocks like [{'type': 'text', 'text': '...'}].

    Args:
        output: Tool output, either a string or list of content blocks.

    Returns:
        Extracted text as a single string.
    """
    if not isinstance(output, list):
        return str(output)
    parts: list[str] = []
    for block in output:
        chunk = (
            block["text"] if isinstance(block, dict) and "text" in block else str(block)
        )
        parts.append(chunk)
    return "\n".join(parts)


def _truncate_tool_text(text: str, tools_token_budget: int) -> tuple[str, bool]:
    """Apply character-level truncation to tool output text.

    A cheap character-level limit (tools_token_budget * 4) is applied
    to avoid the CPU cost of tokenizing arbitrarily large strings.
    The string is cut at the last newline boundary to avoid mid-line splits.

    Args:
        text: Tool output text to truncate.
        tools_token_budget: Remaining token budget; used to derive a
            character limit (~4 chars/token).

    Returns:
        Tuple of (text, was_truncated).
    """
    max_chars = tools_token_budget * _CHARS_PER_TOKEN_ESTIMATE
    if len(text) <= max_chars:
        return text, False
    cut = text[:max_chars].rfind("\n")
    truncated = text[:cut].rstrip("\r") if cut > 0 else text[:max_chars]
    logger.warning(
        "Tool output pre-truncated from %d to %d chars (limit %d)",
        len(text),
        len(truncated),
        max_chars,
    )
    return truncated + _TRUNCATION_WARNING, True


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
    tool: StructuredTool,
    tool_args: dict[str, object],
    tools_token_budget: int,
    offload_manager: "OffloadManager | None" = None,
) -> tuple[str, str, bool, dict | None]:
    """Execute a tool call and return output, status, truncation flag, and structured content.

    Args:
        tool: Tool instance to execute
        tool_args: Arguments to pass to the tool
        tools_token_budget: Remaining token budget for tool outputs
        offload_manager: Optional manager for offloading large outputs to disk

    Returns:
        Tuple of (status, tool_output, was_truncated, structured_content)
    """
    structured_content: dict | None = None
    tool_name = tool.name
    result = await tool.coroutine(**tool_args)  # type: ignore[misc]

    # coroutine may return (content, artifact) tuple for tools with
    # response_format="content_and_artifact", or content directly.
    raw_output = result[0] if isinstance(result, tuple) and len(result) == 2 else result
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        structured_content = result[1].get("structured_content")

    raw_text = _convert_tool_output_to_text(raw_output)

    if offload_manager is not None:
        offloaded = offload_manager.try_offload(raw_text, tool_name, tools_token_budget)
        if offloaded is not raw_text:
            tool_output = offloaded
            was_truncated = False
        else:
            tool_output, was_truncated = _truncate_tool_text(
                raw_text, tools_token_budget
            )
    else:
        tool_output, was_truncated = _truncate_tool_text(raw_text, tools_token_budget)

    status = "success"
    logger.debug(
        "Tool: %s | Args: %s | Output: %s | Truncated: %s | Has structured_content: %s",
        tool_name,
        tool_args,
        tool_output[:200] if len(tool_output) > 200 else tool_output,
        was_truncated,
        structured_content is not None,
    )
    return status, tool_output, was_truncated, structured_content


def _tool_result_event(
    *,
    content: str,
    status: str,
    tool_call_id: str,
    truncated: bool,
    structured_content: dict | None = None,
) -> ToolExecutionEvent:
    """Build a tool_result event payload.

    Args:
        content: Tool output text passed to both LLM and client stream.
        status: Tool execution status value (success/error).
        tool_call_id: Correlation ID of the originating tool call.
        truncated: Whether tool output was truncated due to token limit.
        structured_content: Optional structured data from tool artifact (MCP Apps).

    Returns:
        Tool result event containing a ToolMessage payload.
    """
    additional_kwargs: dict = {"truncated": truncated}
    if structured_content is not None:
        additional_kwargs["structured_content"] = structured_content
    return ToolResultEvent(
        data=ToolMessage(
            content=content,
            status=status,
            tool_call_id=tool_call_id,
            additional_kwargs=additional_kwargs,
        )
    )


def _approval_required_event(
    *,
    approval_id: str,
    tool_name: str,
    tool_description: str,
    tool_args: dict[str, object],
    tool_annotation: dict[str, object],
) -> ApprovalRequiredEvent:
    """Build an approval_required event payload."""
    return ApprovalRequiredEvent(
        data={
            "approval_id": approval_id,
            "tool_name": tool_name,
            "tool_description": tool_description,
            "tool_args": tool_args,
            "tool_annotation": tool_annotation,
        }
    )


def _approval_rejection_event(
    *,
    tool_name: str,
    tool_call_id: str,
    outcome: str,
) -> ToolExecutionEvent:
    """Build non-retryable tool_result event for rejected/timed-out approvals.

    Args:
        tool_name: Name of the tool gated by approval.
        tool_call_id: Correlation ID of the originating tool call.
        outcome: Approval decision outcome (for example "timeout" or "rejected").

    Returns:
        Tool-result event with error status and retry guidance.
    """
    if outcome == "timeout":
        rejection_content = (
            f"Tool '{tool_name}' approval timed out. "
            "Do not retry this exact tool call."
        )
    else:
        rejection_content = (
            f"Tool '{tool_name}' execution was rejected. "
            "Do not retry this exact tool call."
        )
    return _tool_result_event(
        content=rejection_content,
        status="error",
        tool_call_id=tool_call_id,
        truncated=False,
    )


async def _evaluate_and_emit_approval_event(
    *,
    tool_id: str,
    tool: StructuredTool,
    tool_args: dict[str, object],
    streaming: bool,
) -> AsyncGenerator[ToolExecutionEvent, None]:
    """Evaluate approval policy and emit approval events as needed.

    Args:
        tool_id: Correlation ID of the originating tool call.
        tool: Tool being considered for execution.
        tool_args: Tool arguments included in approval-required payloads.
        streaming: Whether this call originated from the streaming endpoint.

    Yields:
        Approval-required event immediately when approval is needed, followed
        by a rejection/timeout tool-result event when approval is not granted.

    Raise:
        _ApprovalNotGrantedError: when approval is explicitly denied or times out.
    """
    tool_name = tool.name

    tool_metadata = tool.metadata if isinstance(tool.metadata, dict) else {}
    tool_annotation = normalize_tool_annotation(tool_metadata)
    need_approval = need_validation(
        streaming=streaming,
        approval_type=config.tools_approval.approval_type,
        tool_annotation=tool_annotation,
    )
    if not need_approval:
        return

    approval_id = str(uuid4())
    register_pending_approval(approval_id=approval_id)
    yield _approval_required_event(
        approval_id=approval_id,
        tool_name=tool_name,
        tool_description=tool.description,
        tool_args=tool_args,
        tool_annotation=tool_annotation,
    )
    outcome = await get_approval_decision(
        approval_id=approval_id,
        timeout_seconds=config.tools_approval.approval_timeout,
    )
    if outcome != "approved":
        yield _approval_rejection_event(
            tool_name=tool_name,
            tool_call_id=tool_id,
            outcome=outcome,
        )
        raise _ApprovalNotGrantedError()


async def _execute_with_retries(
    *,
    tool: StructuredTool,
    tool_args: dict[str, object],
    tools_token_budget: int,
    offload_manager: "OffloadManager | None" = None,
) -> tuple[str, str, bool, dict | None]:
    """Execute one tool call with retry policy.

    Args:
        tool: Tool instance to execute.
        tool_args: Arguments passed to the tool.
        tools_token_budget: Maximum tokens allowed for tool output truncation.
        offload_manager: Optional manager for offloading large outputs to disk.

    Returns:
        Tuple of (status, tool_output, was_truncated, structured_content).
    """
    tool_name = tool.name
    attempts = MAX_TOOL_CALL_RETRIES + 1
    last_error_text = "unknown error"

    for attempt in range(attempts):
        try:
            _status, tool_output, was_truncated, structured_content = (
                await execute_tool_call(
                    tool, tool_args, tools_token_budget, offload_manager
                )
            )
            return "success", tool_output, was_truncated, structured_content
        except Exception as error:
            last_error_text = str(error)
            is_rate_limited_error = _is_rate_limited_tool_error(error)
            should_retry = _is_transient_tool_error(error) or is_rate_limited_error
            if attempt < MAX_TOOL_CALL_RETRIES and should_retry:
                logger.warning(
                    "Retrying tool '%s' after transient error on attempt %d/%d: %s",
                    tool_name,
                    attempt + 1,
                    attempts,
                    error,
                )
                backoff_base = (
                    RATE_LIMIT_RETRY_BACKOFF_SECONDS
                    if is_rate_limited_error
                    else RETRY_BACKOFF_SECONDS
                )
                await asyncio.sleep(backoff_base * (2**attempt))
                continue
            break

    reason = " ".join(last_error_text.split())
    if len(reason) > 220:
        reason = f"{reason[:217]}..."
    tool_output = f"Tool '{tool_name}' failed: {reason}"
    logger.error(tool_output)
    return "error", tool_output, False, None


async def _execute_single_tool_call_stream(
    tool_call: ToolCallDefinition,
    tools_token_budget: int,
    streaming: bool = False,
    offload_manager: "OffloadManager | None" = None,
) -> AsyncGenerator[ToolExecutionEvent, None]:
    """Execute a single tool call and emit execution events.

    Args:
        tool_call: Tuple of (tool_id, tool_args, tool)
        tools_token_budget: Remaining token budget for tool output truncation
        streaming: Whether this call originated from the streaming endpoint
        offload_manager: Optional manager for offloading large outputs to disk

    Yields:
        Approval-required or tool-result events.
    """
    tool_id, tool_args, tool = tool_call

    try:
        async for approval_event in _evaluate_and_emit_approval_event(
            tool_id=tool_id,
            tool=tool,
            tool_args=tool_args,
            streaming=streaming,
        ):
            yield approval_event
    except _ApprovalNotGrantedError:
        return

    status, tool_output, was_truncated, structured_content = (
        await _execute_with_retries(
            tool=tool,
            tool_args=tool_args,
            tools_token_budget=tools_token_budget,
            offload_manager=offload_manager,
        )
    )
    yield _tool_result_event(
        content=tool_output,
        status=status,
        tool_call_id=tool_id,
        truncated=was_truncated,
        structured_content=structured_content,
    )


async def execute_tool_calls_stream(
    tool_calls: list[ToolCallDefinition],
    tools_token_budget: int,
    streaming: bool = False,
    offload_manager: "OffloadManager | None" = None,
) -> AsyncGenerator[ToolExecutionEvent, None]:
    """Execute tool calls in parallel and stream execution events."""
    if not tool_calls:
        return

    per_tool_budget = max(1, tools_token_budget // len(tool_calls))

    # Merge runs all per-tool generators concurrently on the event loop.
    merged = stream.merge(
        *(
            _execute_single_tool_call_stream(
                tc, per_tool_budget, streaming, offload_manager
            )
            for tc in tool_calls
        )
    )
    # Yield events (approval_required / tool_result) as they arrive from any tool.
    async with merged.stream() as streamer:
        async for event in streamer:
            yield event


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
