"""Structured JSON audit logger for compliance audit events."""

import json
import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generator, Optional

from opentelemetry import trace

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _normalize_json(value: Any) -> JsonValue:
    """Coerce a value to a JSON-safe type."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _normalize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    return str(value)


_fallback_logger = logging.getLogger(__name__)


class _AuditJsonFormatter(logging.Formatter):
    """Formatter that passes through pre-formatted JSON strings."""

    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


def _setup_audit_logger() -> logging.Logger:
    """Create and configure the ols.audit logger with JSON output to stdout."""
    logger = logging.getLogger("ols.audit")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_AuditJsonFormatter())
    logger.addHandler(handler)
    return logger


_audit_logger = _setup_audit_logger()


class AuditLogger:
    """Emits structured JSON audit events to stdout.

    When disabled, all methods are no-ops.
    """

    def __init__(self, enabled: bool = True) -> None:
        """Initialize the audit logger."""
        self._enabled = enabled

    def _emit(
        self, event: str, trace_id: str, user_id: str, **fields: JsonValue
    ) -> None:
        if not self._enabled:
            return
        try:
            record: dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "info",
                "event": event,
                "trace_id": trace_id,
                "user_id": user_id,
            }
            record.update({k: _normalize_json(v) for k, v in fields.items()})
            _audit_logger.info(json.dumps(record, ensure_ascii=False, allow_nan=False))
        except Exception:
            _fallback_logger.warning(
                "Failed to emit audit event %s", event, exc_info=True
            )

    def request_started(
        self,
        trace_id: str,
        user_id: str,
        mode: str,
        query: str,
        attachments: list[dict[str, Any]],
        provider: Optional[str],
        model: Optional[str],
    ) -> None:
        """Emit audit.request.started event."""
        self._emit(
            "audit.request.started",
            trace_id,
            user_id,
            mode=mode,
            query=query,
            attachments=attachments,
            provider=provider,
            model=model,
        )

    def request_auth(self, trace_id: str, user_id: str) -> None:
        """Emit audit.request.auth event."""
        self._emit("audit.request.auth", trace_id, user_id)

    def rag_retrieved(
        self,
        trace_id: str,
        user_id: str,
        chunk_count: int,
        scores: list[float],
        source_documents: list[str],
    ) -> None:
        """Emit audit.rag.retrieved event."""
        self._emit(
            "audit.rag.retrieved",
            trace_id,
            user_id,
            chunk_count=chunk_count,
            scores=scores,
            source_documents=source_documents,
        )

    def history_retrieved(
        self,
        trace_id: str,
        user_id: str,
        turn_count: int,
        compressed: bool,
        truncated: bool,
    ) -> None:
        """Emit audit.history.retrieved event."""
        self._emit(
            "audit.history.retrieved",
            trace_id,
            user_id,
            turn_count=turn_count,
            compressed=compressed,
            truncated=truncated,
        )

    def llm_turn(
        self,
        trace_id: str,
        user_id: str,
        turn_index: int,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Emit audit.llm.turn event."""
        self._emit(
            "audit.llm.turn",
            trace_id,
            user_id,
            turn_index=turn_index,
            tokens_in=input_tokens,
            tokens_out=output_tokens,
        )

    def llm_thinking(self, trace_id: str, user_id: str, content: str) -> None:
        """Emit audit.llm.thinking event."""
        self._emit("audit.llm.thinking", trace_id, user_id, content=content)

    def llm_text(self, trace_id: str, user_id: str, content: str) -> None:
        """Emit audit.llm.text event."""
        self._emit("audit.llm.text", trace_id, user_id, content=content)

    def tool_call(
        self,
        trace_id: str,
        user_id: str,
        tool_name: str,
        mcp_server: Optional[str],
        arguments: list[str],
    ) -> None:
        """Emit audit.tool.call event."""
        self._emit(
            "audit.tool.call",
            trace_id,
            user_id,
            tool_name=tool_name,
            mcp_server=mcp_server,
            arguments=arguments,
        )

    def tool_result(
        self,
        trace_id: str,
        user_id: str,
        tool_name: str,
        output_length: int,
        success: bool,
        duration_ms: Optional[int],
    ) -> None:
        """Emit audit.tool.result event."""
        self._emit(
            "audit.tool.result",
            trace_id,
            user_id,
            tool_name=tool_name,
            output_length=output_length,
            success=success,
            duration_ms=duration_ms,
        )

    def tool_approval_requested(
        self,
        trace_id: str,
        user_id: str,
        tool_name: str,
        approval_id: str,
    ) -> None:
        """Emit audit.tool.approval.requested event."""
        self._emit(
            "audit.tool.approval.requested",
            trace_id,
            user_id,
            tool_name=tool_name,
            approval_id=approval_id,
        )

    def tool_approval_decision(
        self,
        trace_id: str,
        user_id: str,
        approval_id: str,
        decision: str,
        tool_name: str,
    ) -> None:
        """Emit audit.tool.approval.decision event."""
        self._emit(
            "audit.tool.approval.decision",
            trace_id,
            user_id,
            approval_id=approval_id,
            decision=decision,
            tool_name=tool_name,
        )

    def request_failed(self, trace_id: str, user_id: str, error: str) -> None:
        """Emit audit.request.failed event."""
        self._emit("audit.request.failed", trace_id, user_id, error=error)

    def request_completed(
        self,
        trace_id: str,
        user_id: str,
        total_turns: int,
        total_input_tokens: int,
        total_output_tokens: int,
        referenced_documents: list[str],
    ) -> None:
        """Emit audit.request.completed event."""
        self._emit(
            "audit.request.completed",
            trace_id,
            user_id,
            total_turns=total_turns,
            total_tokens_in=total_input_tokens,
            total_tokens_out=total_output_tokens,
            referenced_documents=referenced_documents,
        )


@dataclass(frozen=True)
class AuditContext:
    """Immutable context for audit logging throughout a request lifecycle."""

    trace_id: str
    user_id: str
    logger: AuditLogger
    tracer: trace.Tracer = field(default_factory=lambda: trace.get_tracer("ols.audit"))

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Generator[trace.Span, None, None]:
        """Start an OTEL span independent of the logging toggle (per spec rule 3)."""
        with self.tracer.start_as_current_span(name, attributes=attrs) as s:
            yield s
