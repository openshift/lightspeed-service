"""Unit tests for streaming_ols.py."""

import json
from unittest.mock import patch

import pytest

from ols import config, constants

# needs to be setup there before is_user_authorized is imported
config.ols_config.authentication_config.module = "k8s"

from ols.app.endpoints.streaming_ols import (  # noqa:E402
    LLM_HISTORY_COMPRESSION_END_EVENT,
    LLM_HISTORY_COMPRESSION_START_EVENT,
    LLM_REASONING_EVENT,
    LLM_TOKEN_EVENT,
    LLM_TOOL_CALL_EVENT,
    LLM_TOOL_RESULT_EVENT,
    build_referenced_docs,
    format_stream_data,
    generic_llm_error,
    prompt_too_long_error,
    response_processing_wrapper,
    stream_end_event,
    stream_event,
    stream_start_event,
)
from ols.app.models.models import (  # noqa:E402
    LLMRequest,
    RagChunk,
    StreamChunkType,
    StreamedChunk,
    TokenCounter,
)
from ols.utils import suid  # noqa:E402
from ols.utils.errors_parsing import (  # noqa:E402
    _LLM_BACKEND_PREFIX,
    DEFAULT_ERROR_MESSAGE,
)

conversation_id = suid.get_suid()


async def drain_generator(generator) -> str:
    """Drain the async generator and return the result."""
    return [item async for item in generator]


@pytest.fixture(scope="function")
def _load_config():
    """Load config before unit tests."""
    config.reload_from_yaml_file("tests/config/test_app_endpoints.yaml")


def test_event_type_are_not_changed():
    """Test that event types are not changed."""
    assert LLM_TOKEN_EVENT == "token"  # noqa: S105
    assert LLM_REASONING_EVENT == "reasoning"
    assert LLM_TOOL_CALL_EVENT == "tool_call"
    assert LLM_TOOL_RESULT_EVENT == "tool_result"
    assert StreamChunkType.SKILL_SELECTED.value == "skill_selected"
    assert LLM_HISTORY_COMPRESSION_START_EVENT == "history_compression_start"
    assert LLM_HISTORY_COMPRESSION_END_EVENT == "history_compression_end"
    assert StreamChunkType.APPROVAL_REQUIRED.value == "approval_required"


def test_format_stream_data():
    """Test format_stream_data."""
    data = {"bla": 5}
    expected = f"data: {json.dumps(data)}\n\n"
    actual = format_stream_data(data)
    assert actual == expected


def test_stream_event():
    """Test stream_event."""
    data = {"token": "hi", "idx": 1}

    # text output
    assert stream_event(data, LLM_TOKEN_EVENT, constants.MEDIA_TYPE_TEXT) == "hi"
    assert (
        stream_event(data, LLM_TOOL_CALL_EVENT, constants.MEDIA_TYPE_TEXT)
        == '\nTool call: {"token": "hi", "idx": 1}\n'
    )
    assert (
        stream_event(
            data, StreamChunkType.APPROVAL_REQUIRED.value, constants.MEDIA_TYPE_TEXT
        )
        == '\nApproval request: {"token": "hi", "idx": 1}\n'
    )
    assert (
        stream_event(data, LLM_TOOL_RESULT_EVENT, constants.MEDIA_TYPE_TEXT)
        == '\nTool result: {"token": "hi", "idx": 1}\n'
    )
    assert (
        stream_event(
            data, LLM_HISTORY_COMPRESSION_START_EVENT, constants.MEDIA_TYPE_TEXT
        )
        == '\nHistory compression start: {"token": "hi", "idx": 1}\n'
    )
    assert (
        stream_event(data, LLM_HISTORY_COMPRESSION_END_EVENT, constants.MEDIA_TYPE_TEXT)
        == '\nHistory compression end: {"token": "hi", "idx": 1}\n'
    )

    # skill_selected - text shows name, json wraps as SSE
    skill_data = {"name": "pod-diagnostics", "confidence": 0.85}
    assert "pod-diagnostics" in stream_event(
        skill_data, StreamChunkType.SKILL_SELECTED.value, constants.MEDIA_TYPE_TEXT
    )
    assert stream_event(
        skill_data, StreamChunkType.SKILL_SELECTED.value, constants.MEDIA_TYPE_JSON
    ) == (
        'data: {"event": "skill_selected", '
        '"data": {"name": "pod-diagnostics", "confidence": 0.85}}\n\n'
    )

    # skill_selected with missing name falls back to 'unknown'
    assert "unknown" in stream_event(
        {}, StreamChunkType.SKILL_SELECTED.value, constants.MEDIA_TYPE_TEXT
    )

    # json output
    assert (
        stream_event(data, LLM_TOKEN_EVENT, constants.MEDIA_TYPE_JSON)
        == 'data: {"event": "token", "data": {"token": "hi", "idx": 1}}\n\n'
    )
    assert (
        stream_event(data, LLM_TOOL_CALL_EVENT, constants.MEDIA_TYPE_JSON)
        == 'data: {"event": "tool_call", "data": {"token": "hi", "idx": 1}}\n\n'
    )
    assert (
        stream_event(
            data, StreamChunkType.APPROVAL_REQUIRED.value, constants.MEDIA_TYPE_JSON
        )
        == 'data: {"event": "approval_required", "data": {"token": "hi", "idx": 1}}\n\n'
    )
    assert (
        stream_event(data, LLM_TOOL_RESULT_EVENT, constants.MEDIA_TYPE_JSON)
        == 'data: {"event": "tool_result", "data": {"token": "hi", "idx": 1}}\n\n'
    )
    assert (
        stream_event(
            data, LLM_HISTORY_COMPRESSION_START_EVENT, constants.MEDIA_TYPE_JSON
        )
        == 'data: {"event": "history_compression_start", "data": {"token": "hi", "idx": 1}}\n\n'
    )
    assert (
        stream_event(data, LLM_HISTORY_COMPRESSION_END_EVENT, constants.MEDIA_TYPE_JSON)
        == 'data: {"event": "history_compression_end", "data": {"token": "hi", "idx": 1}}\n\n'
    )


def test_stream_event_unknown_type(caplog):
    """Test stream_event with unknown event type."""
    # unknown event
    assert stream_event({}, "bob", constants.MEDIA_TYPE_TEXT) == ""
    assert "Unknown event type: bob" in caplog.text

    # passes through json
    assert (
        stream_event({}, "bob", constants.MEDIA_TYPE_JSON)
        == 'data: {"event": "bob", "data": {}}\n\n'
    )


def test_prompt_too_long_error():
    """Test prompt_too_long_error."""
    assert (
        prompt_too_long_error("error", constants.MEDIA_TYPE_TEXT)
        == "Prompt is too long: error"
    )

    assert prompt_too_long_error(
        "error", constants.MEDIA_TYPE_JSON
    ) == format_stream_data(
        {
            "event": "error",
            "data": {
                "status_code": 413,
                "response": "Prompt is too long",
                "cause": "error",
            },
        }
    )


def test_generic_llm_error():
    """Test generic_llm_error."""
    prefixed_msg = f"{_LLM_BACKEND_PREFIX} {DEFAULT_ERROR_MESSAGE}"
    assert (
        generic_llm_error("error", constants.MEDIA_TYPE_TEXT)
        == f"{prefixed_msg}: error"
    )

    assert generic_llm_error("error", constants.MEDIA_TYPE_JSON) == format_stream_data(
        {
            "event": "error",
            "data": {
                "response": prefixed_msg,
                "cause": "error",
            },
        }
    )


def test_stream_start_event():
    """Test stream_start_event."""
    assert stream_start_event(conversation_id) == format_stream_data(
        {
            "event": "start",
            "data": {
                "conversation_id": conversation_id,
            },
        }
    )


def test_stream_end_event():
    """Test stream_end_event."""
    ref_docs = [{"doc_title": "title_1", "doc_url": "doc_url_1"}]
    truncated = False

    assert (
        stream_end_event(ref_docs, truncated, constants.MEDIA_TYPE_TEXT, None, {})
        == "\n\n---\n\ntitle_1: doc_url_1"
    )

    assert stream_end_event(
        ref_docs, truncated, constants.MEDIA_TYPE_JSON, None, {}
    ) == format_stream_data(
        {
            "event": "end",
            "data": {
                "referenced_documents": [
                    {"doc_title": "title_1", "doc_url": "doc_url_1"}
                ],
                "truncated": truncated,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
            },
            "available_quotas": {},
        }
    )

    token_counter = TokenCounter(
        input_tokens=123, output_tokens=456, reasoning_tokens=78
    )
    assert stream_end_event(
        ref_docs,
        truncated,
        constants.MEDIA_TYPE_JSON,
        token_counter,
        {"limiter1": 10, "limiter2": 20},
    ) == format_stream_data(
        {
            "event": "end",
            "data": {
                "referenced_documents": [
                    {"doc_title": "title_1", "doc_url": "doc_url_1"}
                ],
                "truncated": truncated,
                "input_tokens": 123,
                "output_tokens": 456,
                "reasoning_tokens": 78,
            },
            "available_quotas": {"limiter1": 10, "limiter2": 20},
        }
    )


def test_build_referenced_docs():
    """Test build_referenced_docs."""
    rag_chunks = [
        RagChunk("bla", "url_1", "title_1"),
        RagChunk("bla", "url_2", "title_2"),
        RagChunk("bla", "url_1", "title_1"),  # duplicate
    ]

    assert build_referenced_docs(rag_chunks) == [
        {"doc_title": "title_1", "doc_url": "url_1"},
        {"doc_title": "title_2", "doc_url": "url_2"},
    ]


def test_stream_event_reasoning_text():
    """Test stream_event returns reasoning content for text media type."""
    data = {"reasoning": "thinking step"}
    assert (
        stream_event(data, LLM_REASONING_EVENT, constants.MEDIA_TYPE_TEXT)
        == "thinking step"
    )


def test_stream_event_reasoning_json():
    """Test stream_event wraps reasoning in event envelope for JSON media type."""
    data = {"reasoning": "thinking step"}
    assert stream_event(
        data, LLM_REASONING_EVENT, constants.MEDIA_TYPE_JSON
    ) == format_stream_data(
        {
            "event": "reasoning",
            "data": {"reasoning": "thinking step"},
        }
    )


def test_stream_end_event_with_reasoning_tokens():
    """Test stream_end_event includes reasoning_tokens in JSON output."""
    ref_docs = [{"doc_title": "t", "doc_url": "u"}]
    token_counter = TokenCounter(input_tokens=10, output_tokens=20, reasoning_tokens=30)
    result = stream_end_event(
        ref_docs, False, constants.MEDIA_TYPE_JSON, token_counter, {}
    )
    parsed = json.loads(result.removeprefix("data: ").strip())
    assert parsed["data"]["reasoning_tokens"] == 30
    assert parsed["data"]["input_tokens"] == 10
    assert parsed["data"]["output_tokens"] == 20


@pytest.mark.asyncio
@pytest.mark.usefixtures("_load_config")
async def test_response_processing_wrapper_finalization_error_yields_error_event() -> (
    None
):
    """Verify that a finalization failure yields an error event instead of crashing."""

    async def _fake_generator():
        yield StreamedChunk(type=StreamChunkType.TEXT, text="hello")
        yield StreamedChunk(
            type=StreamChunkType.END,
            data={"rag_chunks": [], "truncated": False, "token_counter": None},
        )

    llm_request = LLMRequest(query="test")

    with patch(
        "ols.app.endpoints.streaming_ols.store_data",
        side_effect=RuntimeError("db connection lost"),
    ):
        events = await drain_generator(
            response_processing_wrapper(
                _fake_generator(),
                user_id="test-user",
                conversation_id=conversation_id,
                llm_request=llm_request,
                attachments=[],
                query_without_attachments="test",
                media_type=constants.MEDIA_TYPE_JSON,
                timestamps={},
                skip_user_id_check=True,
            )
        )

    event_types = []
    error_event = None
    for item in events:
        if item.startswith("data: "):
            parsed = json.loads(item[6:].strip())
            event_types.append(parsed.get("event"))
            if parsed.get("event") == "error":
                error_event = parsed

    assert "start" in event_types
    assert "token" in event_types
    assert "error" in event_types
    assert "end" not in event_types
    assert error_event is not None
    assert "internal error" in error_event["data"]["response"].lower()
    assert "LLM" not in error_event["data"]["response"]
