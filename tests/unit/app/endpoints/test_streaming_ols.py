"""Unit tests for streaming_ols.py."""

import json

import pytest

from ols import config, constants
from ols.app.endpoints.streaming_ols import (
    build_referenced_docs,
    build_yield_item,
    format_stream_data,
    generic_llm_error,
    invalid_response_generator,
    prompt_too_long_error,
    stream_end_event,
    stream_start_event,
)
from ols.app.models.models import RagChunk, TokenCounter
from ols.utils import suid

conversation_id = suid.get_suid()


async def drain_generator(generator) -> str:
    """Drain the async generator and return the result."""
    result = ""
    async for item in generator:
        result += item
    return result


@pytest.fixture(scope="function")
def _load_config():
    """Load config before unit tests."""
    config.reload_from_yaml_file("tests/config/test_app_endpoints.yaml")


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

    assert response == constants.INVALID_QUERY_RESP


def test_build_yield_item():
    """Test build_yield_item."""
    assert build_yield_item("bla", 0, constants.MEDIA_TYPE_TEXT) == "bla"
    assert build_yield_item("bla", 1, constants.MEDIA_TYPE_JSON) == format_stream_data(
        {"event": "token", "data": {"id": 1, "token": "bla"}}
    )


def test_prompt_too_long_error():
    """Test prompt_too_long_error."""
    assert (
        prompt_too_long_error("error", constants.MEDIA_TYPE_TEXT)
        == "Prompt is too long: error"
    )

    assert (
        prompt_too_long_error("error", constants.MEDIA_TYPE_JSON)
        == '{"event": "error", "data": {"status_code": 413, "response": "Prompt is too long", "cause": "error"}}'  # noqa E501
    )


def test_generic_llm_error():
    """Test generic_llm_error."""
    assert (
        generic_llm_error("error", constants.MEDIA_TYPE_TEXT)
        == "Oops, something went wrong during LLM invocation: error"
    )

    assert generic_llm_error("error", constants.MEDIA_TYPE_JSON) == format_stream_data(
        {
            "event": "error",
            "data": {
                "response": "Oops, something went wrong during LLM invocation",
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
        stream_end_event(ref_docs, truncated, constants.MEDIA_TYPE_TEXT, None)
        == "\n\n---\n\ntitle_1: doc_url_1"
    )

    assert stream_end_event(
        ref_docs, truncated, constants.MEDIA_TYPE_JSON, None
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
        }
    )

    token_counter = TokenCounter(input_tokens=123, output_tokens=456)
    assert stream_end_event(
        ref_docs, truncated, constants.MEDIA_TYPE_JSON, token_counter
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
