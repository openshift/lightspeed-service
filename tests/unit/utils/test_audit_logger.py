"""Unit tests for the structured audit logger."""

import json
import logging

import pytest

from ols.utils.audit_logger import AuditContext, AuditLogger


@pytest.fixture()
def captured_records(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Capture JSON strings emitted by the ols.audit logger."""
    records: list[str] = []
    audit_log = logging.getLogger("ols.audit")

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    handler = _Capture()
    monkeypatch.setattr(audit_log, "handlers", [handler])
    return records


class TestAuditLoggerEmit:
    """Verify _emit produces valid JSON with required fields."""

    def test_emit_produces_valid_json(self, captured_records: list[str]) -> None:
        """Test _emit produces valid JSON with all required fields."""
        logger = AuditLogger(enabled=True)
        logger._emit("test.event", "abc123", "user@test", foo="bar")

        assert len(captured_records) == 1
        record = json.loads(captured_records[0])
        assert record["event"] == "test.event"
        assert record["trace_id"] == "abc123"
        assert record["user_id"] == "user@test"
        assert record["foo"] == "bar"
        assert record["level"] == "info"
        assert "timestamp" in record

    def test_disabled_logger_emits_nothing(self, captured_records: list[str]) -> None:
        """Test disabled logger emits no records."""
        logger = AuditLogger(enabled=False)
        logger._emit("test.event", "abc123", "user@test")
        assert len(captured_records) == 0

    def test_emit_nan_does_not_crash(self, captured_records: list[str]) -> None:
        """Test _emit with NaN value logs warning instead of crashing."""
        logger = AuditLogger(enabled=True)
        logger._emit("test.event", "abc123", "user@test", score=float("nan"))
        assert len(captured_records) == 0


class TestAuditLoggerMethods:
    """Verify each typed method emits the correct event name."""

    def test_request_started(self, captured_records: list[str]) -> None:
        """Test request_started emits correct event."""
        logger = AuditLogger()
        logger.request_started(
            "t1",
            "u1",
            mode="chat",
            query="hello",
            attachments=[{"type": "log"}],
            provider="openai",
            model="gpt-4",
        )
        record = json.loads(captured_records[0])
        assert record["event"] == "audit.request.started"
        assert record["mode"] == "chat"
        assert record["query"] == "hello"
        assert record["attachments"] == [{"type": "log"}]
        assert record["provider"] == "openai"
        assert record["model"] == "gpt-4"

    def test_request_auth(self, captured_records: list[str]) -> None:
        """Test request_auth emits correct event."""
        logger = AuditLogger()
        logger.request_auth("t1", "u1")
        record = json.loads(captured_records[0])
        assert record["event"] == "audit.request.auth"

    def test_rag_retrieved(self, captured_records: list[str]) -> None:
        """Test rag_retrieved emits correct event."""
        logger = AuditLogger()
        logger.rag_retrieved(
            "t1", "u1", chunk_count=3, scores=[0.9, 0.8], source_documents=["doc1"]
        )
        record = json.loads(captured_records[0])
        assert record["event"] == "audit.rag.retrieved"
        assert record["chunk_count"] == 3
        assert record["scores"] == [0.9, 0.8]
        assert record["source_documents"] == ["doc1"]

    def test_history_retrieved(self, captured_records: list[str]) -> None:
        """Test history_retrieved emits correct event."""
        logger = AuditLogger()
        logger.history_retrieved(
            "t1", "u1", turn_count=5, compressed=True, truncated=False
        )
        record = json.loads(captured_records[0])
        assert record["event"] == "audit.history.retrieved"
        assert record["turn_count"] == 5
        assert record["compressed"] is True
        assert record["truncated"] is False

    def test_llm_turn(self, captured_records: list[str]) -> None:
        """Test llm_turn emits correct event with token counts."""
        logger = AuditLogger()
        logger.llm_turn("t1", "u1", turn_index=1, input_tokens=100, output_tokens=50)
        record = json.loads(captured_records[0])
        assert record["event"] == "audit.llm.turn"
        assert record["tokens_in"] == 100
        assert record["tokens_out"] == 50

    def test_llm_thinking(self, captured_records: list[str]) -> None:
        """Test llm_thinking emits correct event."""
        logger = AuditLogger()
        logger.llm_thinking("t1", "u1", content="thinking...")
        record = json.loads(captured_records[0])
        assert record["event"] == "audit.llm.thinking"
        assert record["content"] == "thinking..."

    def test_llm_text(self, captured_records: list[str]) -> None:
        """Test llm_text emits correct event."""
        logger = AuditLogger()
        logger.llm_text("t1", "u1", content="response text")
        record = json.loads(captured_records[0])
        assert record["event"] == "audit.llm.text"
        assert record["content"] == "response text"

    def test_tool_call(self, captured_records: list[str]) -> None:
        """Test tool_call emits correct event."""
        logger = AuditLogger()
        logger.tool_call(
            "t1", "u1", tool_name="my_tool", mcp_server="srv", arguments=["a"]
        )
        record = json.loads(captured_records[0])
        assert record["event"] == "audit.tool.call"
        assert record["tool_name"] == "my_tool"
        assert record["mcp_server"] == "srv"
        assert record["arguments"] == ["a"]

    def test_tool_result(self, captured_records: list[str]) -> None:
        """Test tool_result emits correct event."""
        logger = AuditLogger()
        logger.tool_result(
            "t1",
            "u1",
            tool_name="my_tool",
            output_length=2,
            success=True,
            duration_ms=42,
        )
        record = json.loads(captured_records[0])
        assert record["event"] == "audit.tool.result"
        assert record["tool_name"] == "my_tool"
        assert record["output_length"] == 2
        assert record["success"] is True
        assert record["duration_ms"] == 42

    def test_tool_approval_requested(self, captured_records: list[str]) -> None:
        """Test tool_approval_requested emits correct event."""
        logger = AuditLogger()
        logger.tool_approval_requested(
            "t1", "u1", tool_name="my_tool", approval_id="ap1"
        )
        record = json.loads(captured_records[0])
        assert record["event"] == "audit.tool.approval.requested"
        assert record["tool_name"] == "my_tool"
        assert record["approval_id"] == "ap1"

    def test_tool_approval_decision(self, captured_records: list[str]) -> None:
        """Test tool_approval_decision emits correct event."""
        logger = AuditLogger()
        logger.tool_approval_decision(
            "t1", "u1", approval_id="ap1", decision="approved", tool_name="my_tool"
        )
        record = json.loads(captured_records[0])
        assert record["event"] == "audit.tool.approval.decision"
        assert record["approval_id"] == "ap1"
        assert record["decision"] == "approved"
        assert record["tool_name"] == "my_tool"

    def test_request_failed(self, captured_records: list[str]) -> None:
        """Test request_failed emits correct event."""
        logger = AuditLogger()
        logger.request_failed("t1", "u1", error="prompt_too_long")
        record = json.loads(captured_records[0])
        assert record["event"] == "audit.request.failed"
        assert record["error"] == "prompt_too_long"

    def test_request_completed(self, captured_records: list[str]) -> None:
        """Test request_completed emits correct event."""
        logger = AuditLogger()
        logger.request_completed(
            "t1",
            "u1",
            total_turns=3,
            total_input_tokens=500,
            total_output_tokens=200,
            referenced_documents=["https://docs.example.com"],
        )
        record = json.loads(captured_records[0])
        assert record["event"] == "audit.request.completed"
        assert record["total_turns"] == 3
        assert record["total_tokens_in"] == 500
        assert record["total_tokens_out"] == 200
        assert record["referenced_documents"] == ["https://docs.example.com"]

    def test_disabled_methods_are_noop(self, captured_records: list[str]) -> None:
        """Test all methods are no-ops when disabled."""
        logger = AuditLogger(enabled=False)
        logger.request_started(
            "t1", "u1", mode="chat", query="q", attachments=[], provider="p", model="m"
        )
        logger.request_auth("t1", "u1")
        logger.rag_retrieved("t1", "u1", chunk_count=1, scores=[], source_documents=[])
        logger.history_retrieved(
            "t1", "u1", turn_count=0, compressed=False, truncated=False
        )
        logger.llm_turn("t1", "u1", turn_index=1, input_tokens=0, output_tokens=0)
        logger.request_completed(
            "t1",
            "u1",
            total_turns=1,
            total_input_tokens=0,
            total_output_tokens=0,
            referenced_documents=[],
        )
        assert len(captured_records) == 0


class TestAuditContext:
    """Verify AuditContext is frozen and carries the right fields."""

    def test_frozen_dataclass(self) -> None:
        """Test AuditContext is immutable."""
        ctx = AuditContext(trace_id="abc", user_id="user1", logger=AuditLogger())
        assert ctx.trace_id == "abc"
        assert ctx.user_id == "user1"
        with pytest.raises(AttributeError):
            ctx.trace_id = "new"  # type: ignore[misc]


class TestConversationIdToTraceId:
    """Verify UUID to trace ID conversion."""

    def test_strips_hyphens(self) -> None:
        """Test UUID hyphens are stripped to produce 32-char hex."""
        from ols.utils.suid import conversation_id_to_trace_id

        result = conversation_id_to_trace_id("550e8400-e29b-41d4-a716-446655440000")
        assert result == "550e8400e29b41d4a716446655440000"
        assert len(result) == 32

    def test_already_stripped(self) -> None:
        """Test already-stripped IDs pass through unchanged."""
        from ols.utils.suid import conversation_id_to_trace_id

        result = conversation_id_to_trace_id("550e8400e29b41d4a716446655440000")
        assert result == "550e8400e29b41d4a716446655440000"
