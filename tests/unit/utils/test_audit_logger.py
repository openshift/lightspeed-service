"""Unit tests for the structured audit logger (OTel span events)."""

import json

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import SpanKind

from ols.utils.audit_logger import AuditContext, AuditLogger
from ols.utils.otel import OTLPJsonStdoutExporter
from tests.unit.conftest import make_audit_ctx


@pytest.fixture()
def audit_ctx(otel_setup):
    """Create an AuditContext wired to the test provider."""
    return make_audit_ctx(otel_setup, conversation_id="conv-123", user_id="user@test")


def _get_spans(otel_setup):
    """Return collected spans from the otel_setup fixture."""
    exporter, _ = otel_setup
    return exporter.spans


class TestAuditLoggerDisabled:
    """Verify disabled logger emits nothing."""

    def test_disabled_logger_emits_nothing(self, otel_setup) -> None:
        """Verify all methods are no-ops when logger is disabled."""
        exporter, tracer = otel_setup
        ctx = AuditContext(
            conversation_id="c1",
            user_id="u1",
            logger=AuditLogger(enabled=False),
            tracer=tracer,
        )
        with ctx.span("test"):
            ctx.logger.request_started(
                mode="chat", query="q", attachments=[], provider="p", model="m"
            )
            ctx.logger.request_auth()
            ctx.logger.llm_turn(turn_index=1, input_tokens=0, output_tokens=0)
            ctx.logger.request_completed(
                total_turns=1,
                total_input_tokens=0,
                total_output_tokens=0,
                referenced_documents=[],
            )
        spans = exporter.spans
        assert len(spans) == 1
        assert len(spans[0].events) == 0


class TestAuditLoggerMethods:
    """Verify each method emits the correct span event or attributes."""

    def test_request_started(self, audit_ctx, otel_setup) -> None:
        """Verify request_started emits span event with request metadata."""
        with audit_ctx.span("request.lifecycle"):
            audit_ctx.logger.request_started(
                mode="chat",
                query="hello",
                attachments=[{"type": "log"}],
                provider="openai",
                model="gpt-4",
            )
        span = _get_spans(otel_setup)[0]
        assert len(span.events) == 1
        assert span.events[0].name == "request.started"
        assert span.events[0].attributes["mode"] == "chat"
        assert span.events[0].attributes["query"] == "hello"
        assert span.events[0].attributes["provider"] == "openai"

    def test_request_started_no_capture(self, audit_ctx, otel_setup) -> None:
        """Verify request_started omits query when capture_content is False."""
        with audit_ctx.span("request.lifecycle"):
            audit_ctx.logger.request_started(
                mode="chat",
                query="hello",
                attachments=[],
                provider="openai",
                model="gpt-4",
                capture_content=False,
            )
        span = _get_spans(otel_setup)[0]
        assert span.events[0].name == "request.started"
        assert "query" not in span.events[0].attributes

    def test_request_auth_is_noop(self, audit_ctx, otel_setup) -> None:
        """Verify request_auth emits no events."""
        with audit_ctx.span("request.auth"):
            audit_ctx.logger.request_auth()
        assert len(_get_spans(otel_setup)[0].events) == 0

    def test_rag_retrieved(self, audit_ctx, otel_setup) -> None:
        """Verify rag_retrieved sets attributes and adds event."""
        with audit_ctx.span("request.rag"):
            audit_ctx.logger.rag_retrieved(
                chunk_count=3,
                scores=[0.9, 0.8],
                source_documents=["doc1", "doc2"],
            )
        span = _get_spans(otel_setup)[0]
        assert span.attributes["chunk_count"] == 3
        assert span.events[0].name == "rag.retrieved"
        assert span.events[0].attributes["scores"] == (0.9, 0.8)

    def test_history_retrieved(self, audit_ctx, otel_setup) -> None:
        """Verify history_retrieved sets attributes and adds event."""
        with audit_ctx.span("request.history"):
            audit_ctx.logger.history_retrieved(
                turn_count=5, compressed=True, truncated=False
            )
        span = _get_spans(otel_setup)[0]
        assert span.attributes["turn_count"] == 5
        assert span.events[0].name == "history.retrieved"

    def test_llm_turn(self, audit_ctx, otel_setup) -> None:
        """Verify llm_turn sets GenAI token usage attributes."""
        with audit_ctx.span("chat gpt-4", kind=SpanKind.CLIENT):
            audit_ctx.logger.llm_turn(turn_index=1, input_tokens=100, output_tokens=50)
        span = _get_spans(otel_setup)[0]
        assert span.attributes["gen_ai.usage.input_tokens"] == 100
        assert span.attributes["gen_ai.usage.output_tokens"] == 50

    def test_llm_thinking(self, audit_ctx, otel_setup) -> None:
        """Verify llm_thinking emits gen_ai.choice event with reasoning content."""
        with audit_ctx.span("chat gpt-4", kind=SpanKind.CLIENT):
            audit_ctx.logger.llm_thinking(content="thinking...")
        span = _get_spans(otel_setup)[0]
        assert span.events[0].name == "gen_ai.choice"
        assert span.events[0].attributes["gen_ai.reasoning_content"] == "thinking..."

    def test_llm_thinking_no_capture(self, audit_ctx, otel_setup) -> None:
        """Verify llm_thinking omits content when capture_content is False."""
        with audit_ctx.span("chat gpt-4", kind=SpanKind.CLIENT):
            audit_ctx.logger.llm_thinking(content="thinking...", capture_content=False)
        span = _get_spans(otel_setup)[0]
        assert span.events[0].name == "gen_ai.choice"
        assert "gen_ai.reasoning_content" not in span.events[0].attributes

    def test_llm_text(self, audit_ctx, otel_setup) -> None:
        """Verify llm_text emits gen_ai.choice event with completion text."""
        with audit_ctx.span("chat gpt-4", kind=SpanKind.CLIENT):
            audit_ctx.logger.llm_text(content="response text")
        span = _get_spans(otel_setup)[0]
        assert span.events[0].name == "gen_ai.choice"
        assert span.events[0].attributes["gen_ai.completion"] == "response text"

    def test_llm_text_no_capture(self, audit_ctx, otel_setup) -> None:
        """Verify llm_text omits content when capture_content is False."""
        with audit_ctx.span("chat gpt-4", kind=SpanKind.CLIENT):
            audit_ctx.logger.llm_text(content="response text", capture_content=False)
        span = _get_spans(otel_setup)[0]
        assert span.events[0].name == "gen_ai.choice"
        assert "gen_ai.completion" not in span.events[0].attributes

    def test_tool_call(self, audit_ctx, otel_setup) -> None:
        """Verify tool_call emits tool.call span event."""
        with audit_ctx.span("execute_tool my_tool"):
            audit_ctx.logger.tool_call(
                tool_name="my_tool", mcp_server="srv", arguments=["a"]
            )
        span = _get_spans(otel_setup)[0]
        assert span.events[0].name == "tool.call"
        assert span.events[0].attributes["tool_name"] == "my_tool"

    def test_tool_result(self, audit_ctx, otel_setup) -> None:
        """Verify tool_result sets span attributes."""
        with audit_ctx.span("execute_tool my_tool"):
            audit_ctx.logger.tool_result(output_length=2, success=True, duration_ms=42)
        span = _get_spans(otel_setup)[0]
        assert span.attributes["output_length"] == 2
        assert span.attributes["success"] is True
        assert span.attributes["duration_ms"] == 42

    def test_tool_approval_requested(self, audit_ctx, otel_setup) -> None:
        """Verify tool_approval_requested emits approval.requested event."""
        with audit_ctx.span("execute_tool my_tool"):
            audit_ctx.logger.tool_approval_requested(
                tool_name="my_tool", approval_id="ap1"
            )
        assert _get_spans(otel_setup)[0].events[0].name == "approval.requested"

    def test_tool_approval_decision(self, audit_ctx, otel_setup) -> None:
        """Verify tool_approval_decision emits approval.decision event."""
        with audit_ctx.span("execute_tool my_tool"):
            audit_ctx.logger.tool_approval_decision(
                approval_id="ap1", decision="approved", tool_name="my_tool"
            )
        span = _get_spans(otel_setup)[0]
        assert span.events[0].name == "approval.decision"
        assert span.events[0].attributes["decision"] == "approved"

    def test_request_failed(self, audit_ctx, otel_setup) -> None:
        """Verify request_failed sets error status and emits event."""
        with audit_ctx.span("request.lifecycle"):
            audit_ctx.logger.request_failed(error="prompt_too_long")
        span = _get_spans(otel_setup)[0]
        assert span.status.status_code.name == "ERROR"
        assert span.events[0].name == "request.failed"
        assert span.events[0].attributes["error"] == "prompt_too_long"

    def test_request_completed(self, audit_ctx, otel_setup) -> None:
        """Verify request_completed sets attributes and emits event."""
        with audit_ctx.span("request.lifecycle"):
            audit_ctx.logger.request_completed(
                total_turns=3,
                total_input_tokens=500,
                total_output_tokens=200,
                referenced_documents=["https://docs.example.com"],
            )
        span = _get_spans(otel_setup)[0]
        assert span.attributes["total_turns"] == 3
        assert span.attributes["total_input_tokens"] == 500
        assert span.events[0].name == "request.completed"


class TestAuditContext:
    """Verify AuditContext is frozen and carries the right fields."""

    def test_frozen_dataclass(self) -> None:
        """Verify AuditContext is immutable."""
        ctx = AuditContext(
            conversation_id="conv-abc", user_id="user1", logger=AuditLogger()
        )
        assert ctx.conversation_id == "conv-abc"
        assert ctx.user_id == "user1"
        with pytest.raises(AttributeError):
            ctx.conversation_id = "new"  # type: ignore[misc]

    def test_span_injects_conversation_id_and_user_id(self, otel_setup) -> None:
        """Verify span() auto-injects gen_ai.conversation.id and user_id."""
        _, tracer = otel_setup
        ctx = AuditContext(
            conversation_id="conv-xyz",
            user_id="user42",
            logger=AuditLogger(),
            tracer=tracer,
        )
        with ctx.span("test.span"):
            pass
        span = _get_spans(otel_setup)[0]
        assert span.attributes["gen_ai.conversation.id"] == "conv-xyz"
        assert span.attributes["user_id"] == "user42"

    def test_span_kind(self, otel_setup) -> None:
        """Verify span() respects SpanKind parameter."""
        _, tracer = otel_setup
        ctx = AuditContext(
            conversation_id="c1",
            user_id="u1",
            logger=AuditLogger(),
            tracer=tracer,
        )
        with ctx.span("chat gpt-4", kind=SpanKind.CLIENT):
            pass
        assert _get_spans(otel_setup)[0].kind == SpanKind.CLIENT


class TestOTLPJsonStdoutExporter:
    """Verify OTLP JSON stdout exporter produces single-line JSON."""

    def test_exports_single_line_json(self, capsys) -> None:
        """Verify exporter writes valid single-line OTLP JSON to stdout."""
        provider = TracerProvider(
            resource=Resource.create({"service.name": "test-stdout"})
        )
        stdout_exporter = OTLPJsonStdoutExporter()
        provider.add_span_processor(SimpleSpanProcessor(stdout_exporter))

        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("test.span", kind=SpanKind.CLIENT):
            pass

        provider.force_flush()
        captured = capsys.readouterr()
        lines = [x for x in captured.out.strip().split("\n") if x]
        assert len(lines) >= 1
        parsed = json.loads(lines[-1])
        assert "resourceSpans" in parsed
        span = parsed["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert len(span["traceId"]) == 32
        assert all(c in "0123456789abcdef" for c in span["traceId"])
        assert len(span["spanId"]) == 16
        assert all(c in "0123456789abcdef" for c in span["spanId"])
        assert isinstance(span["kind"], int)
        provider.shutdown()
