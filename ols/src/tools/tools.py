"""Functions/Tools definition."""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict
from uuid import uuid4

from langchain_core.messages import ToolMessage
from langchain_core.tools.structured import StructuredTool

from ols import config
from ols.app.models.models import ChunkType
from ols.constants import DEFAULT_MAX_TOKENS_PER_TOOL_OUTPUT
from ols.src.tools.approval import (
    get_approval_decision,
    need_validation,
    normalize_tool_annotation,
    register_pending_approval,
)
from ols.utils.token_handler import TokenHandler

logger = logging.getLogger(__name__)


SENSITIVE_KEYWORDS = ["secret"]
PER_TOOL_EXECUTION_TIMEOUT_SECONDS = 60
MAX_TOOL_CALL_RETRIES = 2
RETRY_BACKOFF_SECONDS = 0.2


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
    event: Literal[ChunkType.APPROVAL_REQUIRED] = ChunkType.APPROVAL_REQUIRED


@dataclass(slots=True)
class ToolResultEvent:
    """Tool-result event emitted during tool execution."""

    data: ToolMessage
    event: Literal[ChunkType.TOOL_RESULT] = ChunkType.TOOL_RESULT


ToolExecutionEvent = ApprovalRequiredEvent | ToolResultEvent
ToolCallDefinition: TypeAlias = tuple[str, dict[str, object], StructuredTool]


def _is_retryable_tool_error(error: Exception) -> bool:
    """Return true if a tool execution error is likely transient."""
    # Canonical timeout/connection exceptions: usually safe to retry.
    if isinstance(error, (TimeoutError, asyncio.TimeoutError, ConnectionError)):
        return True
    # OS/network stack errors (e.g., socket issues) are often transient.
    if isinstance(error, OSError):
        return True
    # Fallback text matching for provider-specific exception types/messages.
    error_text = str(error).lower()
    return any(
        token in error_text
        for token in ("timeout", "temporar", "connection reset", "connection closed")
    )


def _extract_text_from_tool_output(output: object) -> str:
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


def raise_for_sensitive_tool_args(tool_args: dict[str, object]) -> None:
    """Check tool arguments for sensitive content and raise an exception if found."""
    for key, value in tool_args.items():
        if any(keyword in str(value).lower() for keyword in SENSITIVE_KEYWORDS):
            raise ValueError(
                f"Sensitive keyword in tool arguments key '{key}' is not allowed."
            )


async def execute_tool_call(
    tool: StructuredTool,
    tool_args: dict[str, object],
    max_tokens: int = DEFAULT_MAX_TOKENS_PER_TOOL_OUTPUT,
) -> tuple[str, str, bool]:
    """Execute a tool call and return the output, status, and truncation flag.

    Args:
        tool: Tool instance to execute
        tool_args: Arguments to pass to the tool
        max_tokens: Maximum tokens allowed for tool output

    Returns:
        Tuple of (status, tool_output, was_truncated)
    """
    was_truncated = False
    tool_name = tool.name
    result = await asyncio.wait_for(
        tool.ainvoke(_jsonify(tool_args)),
        timeout=PER_TOOL_EXECUTION_TIMEOUT_SECONDS,
    )

    # ainvoke may return (content, artifact) tuple for tools with
    # response_format="content_and_artifact", or content directly.
    # TODO: handle structured_content from artifact for MCP Apps.
    if isinstance(result, tuple) and len(result) == 2:
        tool_output = _extract_text_from_tool_output(result[0])
    else:
        tool_output = _extract_text_from_tool_output(result)

    token_handler = TokenHandler()
    tool_output, was_truncated = token_handler.truncate_tool_output(
        tool_output, max_tokens
    )

    status = "success"
    logger.debug(
        "Tool: %s | Args: %s | Output: %s | Truncated: %s",
        tool_name,
        tool_args,
        tool_output[:500] if len(tool_output) > 500 else tool_output,
        was_truncated,
    )
    return status, tool_output, was_truncated


def _tool_result_event(
    *,
    content: str,
    status: str,
    tool_call_id: str,
    truncated: bool,
) -> ToolExecutionEvent:
    """Build a tool_result event payload.

    Args:
        content: Tool output text passed to both LLM and client stream.
        status: Tool execution status value (success/error).
        tool_call_id: Correlation ID of the originating tool call.
        truncated: Whether tool output was truncated due to token limit.

    Returns:
        Tool result event containing a ToolMessage payload.
    """
    return ToolResultEvent(
        data=ToolMessage(
            content=content,
            status=status,
            tool_call_id=tool_call_id,
            additional_kwargs={"truncated": truncated},
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


async def _execute_single_tool_call_stream(
    tool_call: ToolCallDefinition,
    max_tokens: int = DEFAULT_MAX_TOKENS_PER_TOOL_OUTPUT,
    streaming: bool = False,
) -> AsyncGenerator[ToolExecutionEvent, None]:
    """Execute a single tool call and emit execution events.

    Args:
        tool_call: Tuple of (tool_id, tool_args, tool)
        max_tokens: Maximum tokens allowed for tool output
        streaming: Whether this call originated from the streaming endpoint

    Yields:
        Approval-required or tool-result events.
    """
    tool_id, tool_args, tool = tool_call
    tool_name = tool.name
    was_truncated = False

    try:
        # Pre-execution guard: reject sensitive arguments before any tool side effects.
        raise_for_sensitive_tool_args(tool_args)
    except Exception as error:
        # Return a synthetic non-retryable tool_result to keep LLM flow stable.
        tool_output = (
            f"Tool '{tool_name}' cannot be executed: {error}. "
            "Do not retry this exact tool call."
        )
        logger.exception(tool_output)
        yield _tool_result_event(
            content=tool_output,
            status="error",
            tool_call_id=tool_id,
            truncated=was_truncated,
        )
        return

    # Normalize tool metadata to annotation-only payload for approval policy/event.
    tool_metadata = tool.metadata if isinstance(tool.metadata, dict) else {}
    tool_annotation = normalize_tool_annotation(tool_metadata)
    # Decide whether this specific call requires approval.
    need_approval = need_validation(
        streaming=streaming,
        approval_type=config.tools_approval.approval_type,
        tool_annotation=tool_annotation,
    )
    if need_approval:
        # Generate approval correlation id for this pending approval request.
        approval_id = str(uuid4())
        # Register pending approval before emitting event to avoid decision race.
        register_pending_approval(approval_id=approval_id)
        # Emit approval_required event before waiting so client can prompt user.
        yield _approval_required_event(
            approval_id=approval_id,
            tool_name=tool_name,
            tool_description=tool.description,
            tool_args=tool_args,
            tool_annotation=tool_annotation,
        )
        # Block execution until approval decision or timeout.
        outcome = await get_approval_decision(
            approval_id=approval_id,
            timeout_seconds=config.tools_approval.approval_timeout,
        )
        if outcome != "approved":
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
            # Approval denied/timed out: return synthetic non-retryable result.
            yield _tool_result_event(
                content=rejection_content,
                status="error",
                tool_call_id=tool_id,
                truncated=was_truncated,
            )
            return

    attempts = MAX_TOOL_CALL_RETRIES + 1
    last_error_text = "unknown error"
    # Retry loop for transient failures from actual tool execution.
    for attempt in range(attempts):
        try:
            # Execute tool with timeout + output truncation handling.
            _status, tool_output, was_truncated = await execute_tool_call(
                tool, tool_args, max_tokens
            )
            yield _tool_result_event(
                content=tool_output,
                status="success",
                tool_call_id=tool_id,
                truncated=was_truncated,
            )
            return
        except Exception as error:
            # Capture most recent exception for final non-retryable result message.
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
            break

    tool_output = (
        f"Tool '{tool_name}' execution failed after {attempt + 1} attempt(s): "
        f"{last_error_text}. Do not retry this exact tool call."
    )

    yield _tool_result_event(
        content=tool_output,
        status="error",
        tool_call_id=tool_id,
        truncated=was_truncated,
    )


async def execute_tool_calls_stream(
    tool_calls: list[ToolCallDefinition],
    max_tokens_per_output: int = DEFAULT_MAX_TOKENS_PER_TOOL_OUTPUT,
    streaming: bool = False,
) -> AsyncGenerator[ToolExecutionEvent, None]:
    """Execute tool calls in parallel and stream execution events."""
    # Fast path: nothing to execute, nothing to stream.
    if not tool_calls:
        return

    # Shared fan-in queue for events emitted by all tool workers.
    queue: asyncio.Queue[ToolExecutionEvent | None] = asyncio.Queue()

    # Per-tool worker: execute one tool-call stream and enqueue emitted events.
    async def _run_tool_call(
        tool_call: ToolCallDefinition,
    ) -> None:
        try:
            # Forward each event produced by single-tool execution stream.
            async for event in _execute_single_tool_call_stream(
                tool_call, max_tokens_per_output, streaming
            ):
                await queue.put(event)
        finally:
            # Sentinel marks this worker as finished for consumer bookkeeping.
            await queue.put(None)

    # Launch one worker task per requested tool call.
    tasks = [asyncio.create_task(_run_tool_call(tool_call)) for tool_call in tool_calls]
    # Count how many workers posted completion sentinel.
    finished_workers = 0
    try:
        # Forward events until every worker posts its completion sentinel.
        while finished_workers < len(tasks):
            event = await queue.get()
            if event is None:
                finished_workers += 1
                continue
            # Real tool event from any worker: approval_required or tool_result.
            yield event
    finally:
        # On early exit/cancellation, stop unfinished workers.
        for task in tasks:
            if not task.done():
                task.cancel()
        # Drain task cancellations/exceptions to avoid leaked task warnings.
        await asyncio.gather(*tasks, return_exceptions=True)


def _jsonify(args: dict[str, object]) -> dict[str, object]:
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
