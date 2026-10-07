"""Functions/Tools definition."""

import asyncio
import html
import logging
import re
import time
from collections.abc import AsyncGenerator
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypedDict
from uuid import uuid4

from aiostream import stream
from langchain_core.messages import ToolMessage
from langchain_core.tools.structured import StructuredTool
from opentelemetry.trace import Span, use_span

from ols import config
from ols.app.metrics.metrics import gen_ai_execute_tool_duration_seconds
from ols.app.models.models import StreamChunkType
from ols.src.tools.approval import (
    get_approval_decision,
    need_validation,
    normalize_tool_annotation,
    register_pending_approval,
)
from ols.utils.audit_logger import AuditContext
from ols.utils.token_handler import TokenHandler

if TYPE_CHECKING:
    from ols.src.tools.offloaded_content import OffloadManager

logger = logging.getLogger(__name__)


MAX_TOOL_CALL_RETRIES = 2
RETRY_BACKOFF_SECONDS = 0.2
RATE_LIMIT_RETRY_BACKOFF_SECONDS = 1.0
DO_NOT_RETRY_REMINDER = "Do not retry this exact tool call."

_TRUNCATION_WARNING = (
    "\n[OUTPUT TRUNCATED - The tool returned more data than can be "
    "processed. Please ask a more specific question to get complete results.]"
)
_MINIMAL_TRUNCATION_WARNING = "\n[OUTPUT TRUNCATED]"
_TRUNCATION_WARNING_TOKENS = TokenHandler._get_token_count(
    TokenHandler().text_to_tokens(_TRUNCATION_WARNING)
)
_MINIMAL_TRUNCATION_WARNING_TOKENS = TokenHandler._get_token_count(
    TokenHandler().text_to_tokens(_MINIMAL_TRUNCATION_WARNING)
)
TOOL_RESULT_BUDGET_EXCEEDED_MESSAGE = (
    "Tool results exceeded the remaining token budget. "
    "Please ask a more specific question."
)
INTERNAL_APPROVAL_RESULT_KEY = "ols_internal_approval_result"


class ToolResultBudgetExceededError(Exception):
    """Indicate that the round cannot fit a result or its truncation notice."""


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
    audit_span: Span | None = None


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
    """Convert tool output to plain text without size enforcement."""
    if not isinstance(output, list):
        return str(output)
    parts: list[str] = []
    for block in output:
        chunk = (
            block["text"] if isinstance(block, dict) and "text" in block else str(block)
        )
        parts.append(chunk)
    return "\n".join(parts)


def _truncate_text_to_char_limit(text: str, max_chars: int) -> str:
    """Return up to ``max_chars`` characters, cutting at the last newline when possible."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    prefix = text[:max_chars]
    cut = prefix.rfind("\n")
    chunk = prefix[:cut].rstrip("\r") if cut > 0 else prefix
    return chunk.strip()


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
    Strings are cut at the last newline boundary. For lists, blocks are
    concatenated until the limit; the first block alone may be shortened
    to the limit before appending the truncation notice; later blocks are
    dropped if they would exceed the limit.

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
        body = _truncate_text_to_char_limit(output, max_chars)
        logger.debug(
            "Tool output pre-truncated from %d to %d chars (limit %d)",
            len(output),
            len(body),
            max_chars,
        )
        return body + _TRUNCATION_WARNING, True

    joined = ""
    total = 0
    blocks_kept = 0
    for block in output:
        chunk = (
            block["text"] if isinstance(block, dict) and "text" in block else str(block)
        )
        if total + len(chunk) > max_chars:
            if blocks_kept == 0:
                room = max_chars - total
                body = _truncate_text_to_char_limit(chunk, room)
                logger.debug(
                    "Tool output list: first block exceeds char budget "
                    "(block_len=%d, limit=%d, total_blocks=%d); "
                    "pre-truncating first block to %d chars",
                    len(chunk),
                    max_chars,
                    len(output),
                    len(body),
                )
                return body + _TRUNCATION_WARNING, True
            logger.debug(
                "Tool output pre-truncated at block %d of %d (limit %d chars)",
                blocks_kept,
                len(output),
                max_chars,
            )
            return joined.strip() + _TRUNCATION_WARNING, True
        if blocks_kept > 0:
            joined += "\n" + chunk
        else:
            joined = chunk
        blocks_kept += 1
        total += len(chunk)

    return joined.strip(), False


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
) -> tuple[str, str, bool, dict | None, list | None]:
    """Execute a tool call and return output, status, truncation flag, and metadata.

    Args:
        tool: Tool instance to execute.
        tool_args: Arguments to pass to the tool.
        tools_token_budget: Remaining token budget for tool outputs.
        offload_manager: Optional manager for offloading large outputs to disk.

    Returns:
        Tuple of (status, tool_output, was_truncated, structured_content,
        referenced_documents).
    """
    structured_content: dict | None = None
    referenced_documents: list | None = None
    tool_name = tool.name
    if tool.metadata is not None:
        tool.metadata["tools_token_budget"] = tools_token_budget
    result = await tool.coroutine(**tool_args)  # type: ignore[misc]

    raw_output = result[0] if isinstance(result, tuple) and len(result) == 2 else result
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        raw = result[1].get("structured_content")
        structured_content = raw if isinstance(raw, dict) else None
        raw_refs = result[1].get("referenced_documents")
        referenced_documents = raw_refs if isinstance(raw_refs, list) else None

    if offload_manager is not None:
        raw_text = _convert_tool_output_to_text(raw_output)
        offloaded = offload_manager.try_offload(raw_text, tool_name, tools_token_budget)
        if offloaded is not raw_text:
            tool_output = offloaded
            was_truncated = False
        else:
            tool_output, was_truncated = _extract_text_from_tool_output(
                raw_text, tools_token_budget
            )
    else:
        tool_output, was_truncated = _extract_text_from_tool_output(
            raw_output, tools_token_budget
        )

    status = "success"
    logger.debug(
        "Tool: %s | Args: %s | Output: %s | Truncated: %s | Has structured_content: %s",
        tool_name,
        tool_args,
        tool_output[:200] if len(tool_output) > 200 else tool_output,
        was_truncated,
        structured_content is not None,
    )
    return status, tool_output, was_truncated, structured_content, referenced_documents


def _wrap_tool_output(text: str, tool_name: str) -> str:
    """Mark external tool output as untrusted reference data."""
    escaped_text = re.sub(
        r"</tool_data",
        lambda match: f"<\\/{match.group(0)[2:]}",
        text,
        flags=re.IGNORECASE,
    )
    escaped_name = html.escape(tool_name, quote=True)
    return f'<tool_data source="{escaped_name}">\n{escaped_text}\n</tool_data>'


def _tool_result_event(
    *,
    content: str,
    status: str,
    tool_call_id: str,
    truncated: bool,
    tool_name: str | None = None,
    structured_content: dict | None = None,
    referenced_documents: list | None = None,
    duration_ms: int | None = None,
    audit_span: Span | None = None,
    internal_approval_result: bool = False,
) -> ToolExecutionEvent:
    """Build a tool_result event payload.

    Args:
        content: Raw tool output text before model-facing wrapping.
        status: Tool execution status value (success/error).
        tool_call_id: Correlation ID of the originating tool call.
        truncated: Whether tool output was truncated due to token limit.
        tool_name: Name of the external tool that produced this result.
        structured_content: Optional structured data from tool artifact (MCP Apps).
        referenced_documents: Optional list of RagChunk objects for the API response.
        duration_ms: Wall-clock execution time in milliseconds.
        audit_span: Tool span retained until result inspection completes.
        internal_approval_result: Whether this is a service-generated approval decision.

    Returns:
        Tool result event containing a ToolMessage payload.
    """
    additional_kwargs: dict = {"truncated": truncated}
    if internal_approval_result:
        additional_kwargs[INTERNAL_APPROVAL_RESULT_KEY] = True
    if structured_content is not None:
        additional_kwargs["structured_content"] = structured_content
    if referenced_documents is not None:
        additional_kwargs["referenced_documents"] = referenced_documents
    if duration_ms is not None:
        additional_kwargs["duration_ms"] = duration_ms
    return ToolResultEvent(
        data=ToolMessage(
            content=content,
            status=status,
            tool_call_id=tool_call_id,
            name=tool_name,
            additional_kwargs=additional_kwargs,
        ),
        audit_span=audit_span,
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
    tool_call_id: str,
    outcome: str,
) -> ToolExecutionEvent:
    """Build non-retryable tool_result event for rejected/timed-out approvals.

    Args:
        tool_call_id: Correlation ID of the originating tool call.
        outcome: Approval decision outcome (for example "timeout" or "rejected").

    Returns:
        Tool-result event with error status and retry guidance.
    """
    if outcome == "timeout":
        rejection_content = (
            "Tool approval timed out. Do not retry this exact tool call."
        )
    else:
        rejection_content = (
            "Tool execution was rejected. Do not retry this exact tool call."
        )
    return _tool_result_event(
        content=rejection_content,
        status="error",
        tool_call_id=tool_call_id,
        truncated=False,
        internal_approval_result=True,
    )


async def _evaluate_and_emit_approval_event(
    *,
    tool_id: str,
    tool: StructuredTool,
    tool_args: dict[str, object],
    streaming: bool,
    audit_ctx: AuditContext | None = None,
    audit_span: Span | None = None,
) -> AsyncGenerator[ToolExecutionEvent, None]:
    """Evaluate approval policy and emit approval events as needed.

    Args:
        tool_id: Correlation ID of the originating tool call.
        tool: Tool being considered for execution.
        tool_args: Tool arguments included in approval-required payloads.
        streaming: Whether this call originated from the streaming endpoint.
        audit_ctx: Audit context for structured event logging.
        audit_span: Tool span that owns approval audit events.

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

    outcome: str
    if streaming:
        approval_id = str(uuid4())
        if audit_ctx is None:
            logger.warning(
                "Tool approval requested without audit context; "
                "approval will be unresolvable for tool=%s",
                tool_name,
            )
        user_id = audit_ctx.user_id if audit_ctx else ""
        register_pending_approval(approval_id=approval_id, user_id=user_id)

        if audit_ctx:
            span_context = (
                use_span(audit_span, end_on_exit=False)
                if audit_span is not None
                else nullcontext()
            )
            with span_context:
                audit_ctx.logger.tool_approval_requested(
                    tool_name=tool_name,
                    approval_id=approval_id,
                )

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

        if audit_ctx:
            span_context = (
                use_span(audit_span, end_on_exit=False)
                if audit_span is not None
                else nullcontext()
            )
            with span_context:
                audit_ctx.logger.tool_approval_decision(
                    approval_id=approval_id,
                    decision=outcome,
                    tool_name=tool_name,
                )
    else:
        outcome = "rejected"

    if outcome != "approved":
        yield _approval_rejection_event(
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
) -> tuple[str, str, bool, dict | None, list | None]:
    """Execute one tool call with retry policy.

    Args:
        tool: Tool instance to execute.
        tool_args: Arguments passed to the tool.
        tools_token_budget: Maximum tokens allowed for tool output truncation.
        offload_manager: Optional manager for offloading large outputs to disk.

    Returns:
        Tuple of (status, tool_output, was_truncated, structured_content,
        referenced_documents).
    """
    tool_name = tool.name
    attempts = MAX_TOOL_CALL_RETRIES + 1
    last_error_text = "unknown error"

    for attempt in range(attempts):
        try:
            _status, tool_output, was_truncated, structured_content, ref_docs = (
                await execute_tool_call(
                    tool, tool_args, tools_token_budget, offload_manager
                )
            )
            return "success", tool_output, was_truncated, structured_content, ref_docs
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
    return "error", tool_output, False, None, None


async def _execute_single_tool_call_stream(
    tool_call: ToolCallDefinition,
    tools_token_budget: int,
    streaming: bool = False,
    offload_manager: "OffloadManager | None" = None,
    audit_ctx: AuditContext | None = None,
) -> AsyncGenerator[ToolExecutionEvent, None]:
    """Execute a single tool call and emit execution events.

    Args:
        tool_call: Tuple of (tool_id, tool_args, tool).
        tools_token_budget: Remaining token budget for tool output truncation.
        streaming: Whether this call originated from the streaming endpoint.
        offload_manager: Optional manager for offloading large outputs to disk.
        audit_ctx: Audit context for structured event logging.

    Yields:
        Approval-required or tool-result events.
    """
    tool_id, tool_args, tool = tool_call
    tool_name = getattr(tool, "name", "unknown")
    metadata = getattr(tool, "metadata", None) or {}
    mcp_server = metadata.get("mcp_server", "")

    span_attrs: dict[str, str] = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": tool_name,
        "gen_ai.tool.call.id": tool_id,
        "gen_ai.tool.type": "function",
    }
    if mcp_server:
        span_attrs["mcp.method.name"] = "tools/call"
        mcp_transport = metadata.get("mcp_transport", "")
        if mcp_transport:
            span_attrs["network.transport"] = mcp_transport
        session_id = metadata.get("mcp_session_id", "")
        if session_id:
            span_attrs["mcp.session.id"] = session_id
        protocol_version = metadata.get("mcp_protocol_version", "")
        if protocol_version:
            span_attrs["mcp.protocol.version"] = protocol_version

    tool_span = (
        audit_ctx.start_span(
            f"execute_tool {tool_name}", **span_attrs, mcp_server=mcp_server
        )
        if audit_ctx
        else None
    )
    result_span_transferred = False
    try:
        if audit_ctx and tool_span:
            tool_call_span_context = use_span(tool_span, end_on_exit=False)
            with tool_call_span_context:  # pylint: disable=not-context-manager
                audit_ctx.logger.tool_call(
                    tool_name=tool_name,
                    mcp_server=mcp_server or None,
                    arguments=list(tool_args.keys()),
                )

        try:
            async for approval_event in _evaluate_and_emit_approval_event(
                tool_id=tool_id,
                tool=tool,
                tool_args=tool_args,
                streaming=streaming,
                audit_ctx=audit_ctx,
                audit_span=tool_span,
            ):
                yield approval_event
        except _ApprovalNotGrantedError:
            return

        t0 = time.monotonic()
        span_context = (
            use_span(tool_span, end_on_exit=False) if tool_span else nullcontext()
        )
        with span_context:
            status, tool_output, was_truncated, structured_content, ref_docs = (
                await _execute_with_retries(
                    tool=tool,
                    tool_args=tool_args,
                    tools_token_budget=tools_token_budget,
                    offload_manager=offload_manager,
                )
            )
        elapsed = time.monotonic() - t0
        duration_ms = int(elapsed * 1000)
        gen_ai_execute_tool_duration_seconds.labels(
            gen_ai_tool_name=tool_name,
        ).observe(elapsed)

        event = _tool_result_event(
            content=tool_output,
            status=status,
            tool_call_id=tool_id,
            truncated=was_truncated,
            structured_content=structured_content,
            referenced_documents=ref_docs,
            duration_ms=duration_ms,
            tool_name=tool_name,
            audit_span=tool_span,
        )
        result_span_transferred = tool_span is not None
        yield event
    finally:
        if (
            tool_span is not None
            and not result_span_transferred
            and tool_span.is_recording()
        ):
            tool_span.end()


async def execute_tool_calls_stream(
    tool_calls: list[ToolCallDefinition],
    tools_token_budget: int,
    streaming: bool = False,
    offload_manager: "OffloadManager | None" = None,
    audit_ctx: AuditContext | None = None,
) -> AsyncGenerator[ToolExecutionEvent, None]:
    """Execute tool calls in parallel and stream execution events."""
    if not tool_calls:
        return

    per_tool_budget = max(1, tools_token_budget // len(tool_calls))

    # Merge runs all per-tool generators concurrently on the event loop.
    merged = stream.merge(
        *(
            _execute_single_tool_call_stream(
                tc, per_tool_budget, streaming, offload_manager, audit_ctx
            )
            for tc in tool_calls
        )
    )
    # Yield events (approval_required / tool_result) as they arrive from any tool.
    unclaimed_audit_spans: set[Span] = set()
    try:
        async with merged.stream() as streamer:
            async for event in streamer:
                if isinstance(event, ToolResultEvent) and event.audit_span is not None:
                    unclaimed_audit_spans.add(event.audit_span)
                yield event
                if isinstance(event, ToolResultEvent) and event.audit_span is not None:
                    unclaimed_audit_spans.discard(event.audit_span)
    finally:
        for audit_span in unclaimed_audit_spans:
            if audit_span.is_recording():
                audit_span.end()


def _tool_message_budget_content(message: ToolMessage) -> str:
    content = str(message.content)
    if message.name is None:
        return content
    return _wrap_tool_output(content, message.name)


def _truncate_wrapped_tool_content(
    content: str,
    tool_name: str,
    token_limit: int,
    token_handler: TokenHandler,
) -> tuple[str, int, bool]:
    content_tokens = token_handler.text_to_tokens(content)
    warning = _TRUNCATION_WARNING
    warning_tokens = TokenHandler._get_token_count(
        token_handler.text_to_tokens(_wrap_tool_output(warning, tool_name))
    )
    if warning_tokens > token_limit:
        warning = _MINIMAL_TRUNCATION_WARNING
        warning_tokens = TokenHandler._get_token_count(
            token_handler.text_to_tokens(_wrap_tool_output(warning, tool_name))
        )
    if warning_tokens > token_limit:
        return "", 0, False

    content_limit = max(0, token_limit - warning_tokens)

    while True:
        raw = token_handler.tokens_to_text(content_tokens[:content_limit])
        cut = raw.rfind("\n")
        body = raw[:cut].rstrip("\r") if cut > 0 else raw
        truncated_text = body.strip() + warning
        wrapped_count = TokenHandler._get_token_count(
            token_handler.text_to_tokens(_wrap_tool_output(truncated_text, tool_name))
        )
        if wrapped_count <= token_limit:
            return truncated_text, wrapped_count, True
        if content_limit == 0:
            return warning, warning_tokens, True
        content_limit = max(0, content_limit - (wrapped_count - token_limit))


def _truncate_unwrapped_tool_content(
    content_tokens: list[int], token_limit: int, token_handler: TokenHandler
) -> tuple[str, int]:
    """Truncate generated content to its exact token limit."""
    warning = _TRUNCATION_WARNING
    warning_tokens = _TRUNCATION_WARNING_TOKENS
    if warning_tokens > token_limit:
        warning = _MINIMAL_TRUNCATION_WARNING
        warning_tokens = _MINIMAL_TRUNCATION_WARNING_TOKENS
    if warning_tokens > token_limit:
        return "", 0

    content_limit = max(0, token_limit - warning_tokens)
    while True:
        raw = token_handler.tokens_to_text(content_tokens[:content_limit])
        cut = raw.rfind("\n")
        body = raw[:cut].rstrip("\r") if cut > 0 else raw
        truncated_text = body.strip() + warning
        token_count = TokenHandler._get_token_count(
            token_handler.text_to_tokens(truncated_text)
        )
        if token_count <= token_limit:
            return truncated_text, token_count
        if content_limit == 0:
            return warning, warning_tokens
        content_limit = max(0, content_limit - (token_count - token_limit))


def _truncate_tool_message(
    message: ToolMessage,
    token_list: list[int],
    token_limit: int,
    token_handler: TokenHandler,
) -> tuple[ToolMessage, int]:
    """Truncate one result while keeping its model-facing representation in budget."""
    result_name = message.name
    if message.name is None:
        truncated_text, token_count = _truncate_unwrapped_tool_content(
            token_list, token_limit, token_handler
        )
    else:
        truncated_text, token_count, can_keep_wrapper = _truncate_wrapped_tool_content(
            str(message.content), message.name, token_limit, token_handler
        )
        if not can_keep_wrapper:
            raise ToolResultBudgetExceededError(TOOL_RESULT_BUDGET_EXCEEDED_MESSAGE)

    truncated_message = ToolMessage(
        content=truncated_text,
        status=message.status,
        tool_call_id=message.tool_call_id,
        name=result_name,
        additional_kwargs={
            **message.additional_kwargs,
            "truncated": True,
            "token_count": token_count,
        },
    )
    return truncated_message, token_count


def _minimum_tool_message_token_count(
    message: ToolMessage, token_count: int, token_handler: TokenHandler
) -> int:
    """Return the smallest representation that explains a truncated result."""
    if message.name is None:
        minimum_content = _MINIMAL_TRUNCATION_WARNING
    else:
        minimum_content = _wrap_tool_output(_MINIMAL_TRUNCATION_WARNING, message.name)
    minimum_count = TokenHandler._get_token_count(
        token_handler.text_to_tokens(minimum_content)
    )
    return min(token_count, minimum_count)


def _allocate_limits_with_minimums(
    token_counts: list[int], minimum_counts: list[int], remaining_budget: int
) -> list[int]:
    """Allocate a round budget while reserving a visible notice per truncated result."""
    minimum_total = sum(minimum_counts)
    if minimum_total > remaining_budget:
        raise ToolResultBudgetExceededError(TOOL_RESULT_BUDGET_EXCEEDED_MESSAGE)

    capacities = [
        count - minimum for count, minimum in zip(token_counts, minimum_counts)
    ]
    remaining = remaining_budget - minimum_total
    total_capacity = sum(capacities)
    if total_capacity == 0 or remaining == 0:
        return minimum_counts

    allocations = [
        minimum + (remaining * capacity // total_capacity)
        for minimum, capacity in zip(minimum_counts, capacities)
    ]
    unassigned = remaining_budget - sum(allocations)
    for index, capacity in enumerate(capacities):
        if unassigned == 0:
            break
        if allocations[index] < token_counts[index] and capacity > 0:
            allocations[index] += 1
            unassigned -= 1
    return allocations


def enforce_tool_token_budget(
    tool_messages: list[ToolMessage],
    remaining_budget: int,
    token_handler: TokenHandler,
) -> list[ToolMessage]:
    """Ensure combined tool outputs fit within the remaining token budget.

    Uses a three-tier strategy to avoid unnecessary tokenization:
    1. Cheap character-based estimate — skip tokenization if clearly under budget.
    2. Precise tokenization — only if the estimate suggests overflow.
    3. Truncation — preserve a visible notice for every truncated result and
       reallocate the round budget when a result's share is too small.

    Raises:
        ToolResultBudgetExceededError: If the round cannot fit a truncation notice
            for every result that must be truncated.

    Args:
        tool_messages: Tool result messages to enforce budget on.
        remaining_budget: Remaining token budget for tool outputs.
        token_handler: Tokenizer to use for token counting and truncation.

    Returns:
        The same list of ToolMessages, with oversized ones truncated in place.
    """
    if not tool_messages:
        return tool_messages

    # Tier 1: cheap char-based estimate (~4 chars/token), with exact wrapper
    # overhead reserved for each named result. The 0.9 factor compensates for
    # the character approximation; if we're clearly under budget, skip the
    # expensive tokenization entirely.
    estimated_tokens = sum(
        (
            TokenHandler._get_token_count(
                token_handler.text_to_tokens(_wrap_tool_output("", msg.name))
            )
            if msg.name is not None
            else 0
        )
        + len(str(msg.content)) // _CHARS_PER_TOKEN_ESTIMATE
        for msg in tool_messages
    )
    if estimated_tokens <= int(remaining_budget * 0.9):
        return tool_messages

    # Tier 2: precise tokenization. The char estimate was ambiguous,
    # so tokenize each message to get exact counts.
    token_lists = [
        token_handler.text_to_tokens(_tool_message_budget_content(msg))
        for msg in tool_messages
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
    minimum_counts = [
        _minimum_tool_message_token_count(message, count, token_handler)
        for message, count in zip(tool_messages, token_counts)
    ]

    # If the longest result can absorb the excess, keep the other results intact
    # unless doing so would leave no room for the truncation notice.
    if token_counts[longest_idx] // 2 >= excess:
        longest_limit = token_counts[longest_idx] - excess
        if longest_limit >= minimum_counts[longest_idx]:
            targets = [longest_idx]
            limits = [longest_limit]
        else:
            targets = list(range(len(token_counts)))
            limits = _allocate_limits_with_minimums(
                token_counts, minimum_counts, remaining_budget
            )
    else:
        targets = list(range(len(token_counts)))
        limits = _allocate_limits_with_minimums(
            token_counts, minimum_counts, remaining_budget
        )
        logger.debug(
            "Scaling all %d messages to fit budget %d (total %d)",
            len(targets),
            remaining_budget,
            total,
        )

    # Truncate using pre-computed tokens to avoid re-tokenization.
    for idx, limit in zip(targets, limits):
        if token_counts[idx] <= limit:
            continue
        tool_messages[idx], token_counts[idx] = _truncate_tool_message(
            tool_messages[idx], token_lists[idx], limit, token_handler
        )

    return tool_messages
