"""Unit tests for streaming_ols.py."""

import json

import pytest

from ols import config, constants
from ols.app.models.models import ChunkType, StreamedChunk

# needs to be setup there before is_user_authorized is imported
config.ols_config.authentication_config.module = "k8s"

from ols.app.endpoints.streaming_ols import (  # noqa:E402
    TOKEN_KEY_TOKEN,
    build_referenced_docs,
    format_stream_data,
    generic_llm_error,
    invalid_response_generator,
    prompt_too_long_error,
    stream_end_event,
    stream_event,
    stream_start_event,
)
from ols.app.models.models import RagChunk, TokenCounter  # noqa:E402
from ols.customize import prompts  # noqa:E402
from ols.utils import suid  # noqa:E402
from ols.utils.errors_parsing import DEFAULT_ERROR_MESSAGE  # noqa:E402

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
    assert TOKEN_KEY_TOKEN == "token"  # noqa: S105
    assert ChunkType.TOOL_CALL.value == "tool_call"
    assert ChunkType.APPROVAL_REQUIRED.value == "approval_required"
    assert ChunkType.TOOL_RESULT.value == "tool_result"


def test_format_stream_data():
    """Test format_stream_data."""
    data = {"bla": 5}
    expected = f"data: {json.dumps(data)}\n\n"
    actual = format_stream_data(data)
    assert actual == expected


@pytest.mark.asyncio
@pytest.mark.usefixtures("_load_config")
async def test_invalid_response_generator():
    """Test invalid_response_generator."""
    generator = invalid_response_generator()

    response = await drain_generator(generator)

    assert len(response) == 1
    assert isinstance(response[0], StreamedChunk)
    assert response[0].text == prompts.INVALID_QUERY_RESP


def test_stream_event():
    """Test stream_event."""
    data = {"token": "hi", "idx": 1}

    # text output
    assert stream_event(data, TOKEN_KEY_TOKEN, constants.MEDIA_TYPE_TEXT) == "hi"
    assert (
        stream_event(data, ChunkType.TOOL_CALL.value, constants.MEDIA_TYPE_TEXT)
        == '\nTool call: {"token": "hi", "idx": 1}\n'
    )
    assert (
        stream_event(data, ChunkType.APPROVAL_REQUIRED.value, constants.MEDIA_TYPE_TEXT)
        == '\nApproval request: {"token": "hi", "idx": 1}\n'
    )
    assert (
        stream_event(data, ChunkType.TOOL_RESULT.value, constants.MEDIA_TYPE_TEXT)
        == '\nTool result: {"token": "hi", "idx": 1}\n'
    )

    # json output
    assert (
        stream_event(data, TOKEN_KEY_TOKEN, constants.MEDIA_TYPE_JSON)
        == 'data: {"event": "token", "data": {"token": "hi", "idx": 1}}\n\n'
    )
    assert (
        stream_event(data, ChunkType.TOOL_CALL.value, constants.MEDIA_TYPE_JSON)
        == 'data: {"event": "tool_call", "data": {"token": "hi", "idx": 1}}\n\n'
    )
    assert (
        stream_event(data, ChunkType.APPROVAL_REQUIRED.value, constants.MEDIA_TYPE_JSON)
        == 'data: {"event": "approval_required", "data": {"token": "hi", "idx": 1}}\n\n'
    )
    assert (
        stream_event(data, ChunkType.TOOL_RESULT.value, constants.MEDIA_TYPE_JSON)
        == 'data: {"event": "tool_result", "data": {"token": "hi", "idx": 1}}\n\n'
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
    assert (
        generic_llm_error("error", constants.MEDIA_TYPE_TEXT)
        == f"{DEFAULT_ERROR_MESSAGE}: error"
    )

    assert generic_llm_error("error", constants.MEDIA_TYPE_JSON) == format_stream_data(
        {
            "event": "error",
            "data": {
                "response": DEFAULT_ERROR_MESSAGE,
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
            },
            "available_quotas": {},
        }
    )

    token_counter = TokenCounter(input_tokens=123, output_tokens=456)
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
