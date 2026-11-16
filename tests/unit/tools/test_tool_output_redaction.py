"""Tool execution redaction boundary regression tests."""

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.tools import StructuredTool

from ols import config
from ols.src.redaction.engine import Redactor
from ols.src.tools.offloaded_content import OffloadManager
from ols.src.tools.tool_result_inspection import ToolResultInspectionError

config.ols_config.authentication_config.module = "k8s"

from ols.src.tools.tools import ToolResultEvent, execute_tool_calls_stream  # noqa: E402


async def _run(output, *, budget=4000, error=False, offload_manager=None):
    async def execute():
        if error:
            raise ValueError(output)
        return output

    tool = StructuredTool.from_function(
        coroutine=execute, name="external", description="Return a test tool result"
    )
    events = [
        event
        async for event in execute_tool_calls_stream(
            [("call-1", {}, tool)], budget, offload_manager=offload_manager
        )
        if isinstance(event, ToolResultEvent)
    ]
    return events[0].data


@pytest.mark.asyncio
async def test_tool_result_redacted_before_delivery():
    """Return sanitized text with explicit metadata only when values changed."""
    message = await _run("contact admin@example.com")
    assert message.content == "contact [REDACTED:email]"
    assert message.additional_kwargs["redacted"] is True
    assert message.additional_kwargs["redacted_count"] == 1
    assert message.additional_kwargs["redaction_counts"] == {"email": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "enabled, text", [(True, "benign output"), (False, "admin@example.com")]
)
async def test_unchanged_results_have_no_redaction_metadata(enabled, text):
    """Omit UI indicators when there are no replacements, including explicit opt-out."""
    config.ols_config.tool_output_redaction = enabled
    message = await _run(text)
    assert message.content == text
    assert "redacted" not in message.additional_kwargs
    assert "redacted_count" not in message.additional_kwargs
    assert "redaction_counts" not in message.additional_kwargs


@pytest.mark.asyncio
async def test_tool_error_is_redacted():
    """Sanitize external error text without retrying a redaction failure."""
    message = await _run("contact admin@example.com", error=True)
    assert message.status == "error"
    assert "admin@example.com" not in message.content
    assert message.additional_kwargs["redacted"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [False, True])
async def test_tool_redaction_runs_after_size_guard(error):
    """Run regexes only over the bounded result, not the huge original output."""
    original = "safe line\n" * 10000 + "admin@example.com"
    with patch.object(Redactor, "redact", wraps=Redactor.redact) as redact:
        message = await _run(original, budget=100, error=error)
    if not error:
        assert message.additional_kwargs["truncated"] is True
    assert max(len(call.args[0]) for call in redact.call_args_list) < 1000


@pytest.mark.asyncio
async def test_redaction_failure_does_not_forward_raw_result(caplog):
    """Fail closed with a controlled exception, without leaking engine exception text."""
    with patch.object(
        Redactor, "redact", side_effect=TimeoutError("private engine error")
    ) as redact:
        with pytest.raises(
            ToolResultInspectionError, match="Tool-output redaction failed"
        ) as caught:
            await _run("admin@example.com")
    assert redact.call_count == 1
    assert "private engine error" not in str(caught.value)
    assert "private engine error" not in caplog.text
    assert "admin@example.com" not in caplog.text


@pytest.mark.asyncio
async def test_structured_projection_counts_each_replacement():
    """Sum replacements performed in text and structured content."""
    message = await _run(
        ("admin@example.com", {"structured_content": {"email": "admin@example.com"}})
    )
    assert message.additional_kwargs["structured_content"] == {
        "email": "[REDACTED:email]"
    }
    assert message.additional_kwargs["redacted_count"] == 2
    assert message.additional_kwargs["redaction_counts"] == {"email": 2}


@pytest.mark.asyncio
@pytest.mark.parametrize("write_fails", [False, True])
async def test_non_offloaded_results_keep_structured_content(tmp_path, write_fails):
    """Preserve structured data for small results and failed-offload fallbacks."""
    manager = OffloadManager(storage_path=str(tmp_path))
    raw = "safe line\n" * 1000 if write_fails else "safe line"
    try:
        with patch(
            "ols.src.tools.offloaded_content.os.open",
            side_effect=OSError("test write failure"),
        ):
            message = await _run(
                (raw, {"structured_content": {"email": "admin@example.com"}}),
                budget=100,
                offload_manager=manager,
            )
        assert not manager.has_offloaded_content
        assert message.additional_kwargs["truncated"] is write_fails
        assert message.additional_kwargs["structured_content"] == {
            "email": "[REDACTED:email]"
        }
    finally:
        manager.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_name", ["read_offloaded_content", "search_offloaded_content"]
)
async def test_offload_redaction_on_read_not_write(tmp_path, tool_name, caplog):
    """Omit the huge structured projection and sanitize disk content on retrieval."""
    caplog.set_level("DEBUG")
    manager = OffloadManager(storage_path=str(tmp_path))
    raw = "contact admin@example.com\n" * 1000
    try:
        initial = await _run(
            (raw, {"structured_content": {"rows": raw}}),
            budget=100,
            offload_manager=manager,
        )
        assert initial.content.startswith("[Offloaded:")
        assert "structured_content" not in initial.additional_kwargs
        assert "redacted" not in initial.additional_kwargs
        ref_id = next(iter(manager._allowlist))
        assert (
            await asyncio.to_thread(Path(manager._allowlist[ref_id]).read_text) == raw
        )
        tool = next(
            tool for tool in manager.build_retrieval_tools() if tool.name == tool_name
        )
        args = {"ref_id": ref_id}
        if tool_name == "read_offloaded_content":
            args.update(start_line=1, end_line=2)
        else:
            args.update(pattern="contact", context_lines=0)
        events = [
            event
            async for event in execute_tool_calls_stream(
                [("read-1", args, tool)], 100, offload_manager=manager
            )
            if isinstance(event, ToolResultEvent)
        ]
        result = events[0].data
        assert "admin@example.com" not in result.content
        assert "[REDACTED:email]" in result.content
        assert result.additional_kwargs["redacted"] is True
        assert "admin@example.com" not in caplog.text
        assert (
            await asyncio.to_thread(Path(manager._allowlist[ref_id]).read_text) == raw
        )
    finally:
        manager.cleanup()
