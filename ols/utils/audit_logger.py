"""Structured audit logger emitting OTel span events for compliance."""

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode

_fallback_logger = logging.getLogger(__name__)


class AuditLogger:
    """Emits audit data as OTel span events.

    When disabled, all methods are no-ops.
    """

    def __init__(self, enabled: bool = True) -> None:
        """Initialize the audit logger."""
        self._enabled = enabled

    def _add_span_event(self, name: str, **attrs: Any) -> None:
        if not self._enabled:
            return
        span = trace.get_current_span()
        if span is None or not span.is_recording():
            return
        filtered = {k: v for k, v in attrs.items() if v is not None}
        span.add_event(name, attributes=filtered)

    def _set_span_attrs(self, **attrs: Any) -> None:
        if not self._enabled:
            return
        span = trace.get_current_span()
        if span is None or not span.is_recording():
            return
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(k, v)

    def request_started(
        self,
        *,
        mode: str,
        query: str,
        attachments: list[dict[str, Any]],
        provider: Optional[str],
        model: Optional[str],
        capture_content: bool = True,
    ) -> None:
        """Emit request.started span event on request.lifecycle span."""
        self._add_span_event(
            "request.started",
            mode=mode,
            query=query if capture_content else None,
            attachment_count=len(attachments),
            provider=provider or "",
            model=model or "",
        )

    def request_auth(self) -> None:
        """No-op — auth is tracked by its own span."""

    def rag_retrieved(
        self,
        *,
        chunk_count: int,
        scores: list[float],
        source_documents: list[str],
    ) -> None:
        """Set RAG attributes on request.rag span."""
        self._set_span_attrs(chunk_count=chunk_count)
        self._add_span_event(
            "rag.retrieved",
            chunk_count=chunk_count,
            scores=[round(s, 4) for s in scores],
            source_documents=source_documents,
        )

    def history_retrieved(
        self,
        *,
        turn_count: int,
        compressed: bool,
        truncated: bool,
    ) -> None:
        """Set history attributes on request.history span."""
        self._set_span_attrs(turn_count=turn_count)
        self._add_span_event(
            "history.retrieved",
            turn_count=turn_count,
            compressed=compressed,
            truncated=truncated,
        )

    def llm_turn(
        self,
        *,
        turn_index: int,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Set token usage attributes on the chat span."""
        self._set_span_attrs(
            **{
                "gen_ai.usage.input_tokens": input_tokens,
                "gen_ai.usage.output_tokens": output_tokens,
                "turn_index": turn_index,
            }
        )

    def llm_thinking(self, *, content: str, capture_content: bool = True) -> None:
        """Emit gen_ai.choice span event for reasoning content."""
        attrs: dict[str, Any] = {}
        if capture_content:
            attrs["gen_ai.reasoning_content"] = content
        self._add_span_event("gen_ai.choice", **attrs)

    def llm_text(self, *, content: str, capture_content: bool = True) -> None:
        """Emit gen_ai.choice span event for completion text."""
        attrs: dict[str, Any] = {}
        if capture_content:
            attrs["gen_ai.completion"] = content
        self._add_span_event("gen_ai.choice", **attrs)

    def tool_call(
        self,
        *,
        tool_name: str,
        mcp_server: Optional[str],
        arguments: list[str],
    ) -> None:
        """Emit tool.call span event on execute_tool span."""
        self._add_span_event(
            "tool.call",
            tool_name=tool_name,
            mcp_server=mcp_server or "",
            arguments=arguments,
        )

    def tool_result(
        self,
        *,
        output_length: int,
        success: bool,
        duration_ms: Optional[int],
    ) -> None:
        """Set tool result attributes on execute_tool span."""
        self._set_span_attrs(
            output_length=output_length,
            success=success,
            duration_ms=duration_ms,
        )

    def tool_approval_requested(
        self,
        *,
        tool_name: str,
        approval_id: str,
    ) -> None:
        """Emit approval.requested span event."""
        self._add_span_event(
            "approval.requested",
            tool_name=tool_name,
            approval_id=approval_id,
        )

    def tool_approval_decision(
        self,
        *,
        approval_id: str,
        decision: str,
        tool_name: str,
    ) -> None:
        """Emit approval.decision span event."""
        self._add_span_event(
            "approval.decision",
            approval_id=approval_id,
            decision=decision,
            tool_name=tool_name,
        )

    def request_failed(self, *, error: str) -> None:
        """Set error status and event on request.lifecycle span."""
        if not self._enabled:
            return
        span = trace.get_current_span()
        if span is None or not span.is_recording():
            return
        span.set_status(StatusCode.ERROR, error)
        span.add_event("request.failed", attributes={"error": error})

    def request_completed(
        self,
        *,
        total_turns: int,
        total_input_tokens: int,
        total_output_tokens: int,
        referenced_documents: list[str],
    ) -> None:
        """Set completion attributes on request.lifecycle span."""
        self._set_span_attrs(
            total_turns=total_turns,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
        )
        self._add_span_event(
            "request.completed",
            total_turns=total_turns,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            referenced_documents=referenced_documents,
        )


@dataclass(frozen=True)
class AuditContext:
    """Immutable context for audit logging throughout a request lifecycle."""

    conversation_id: str
    user_id: str
    logger: AuditLogger
    capture_content: bool = True
    tracer: trace.Tracer = field(default_factory=lambda: trace.get_tracer("ols.audit"))

    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        **attrs: Any,
    ) -> Generator[trace.Span, None, None]:
        """Start an OTEL span with conversation_id and user_id auto-injected."""
        attrs.setdefault("gen_ai.conversation.id", self.conversation_id)
        attrs.setdefault("user_id", self.user_id)
        with self.tracer.start_as_current_span(name, kind=kind, attributes=attrs) as s:
            yield s
