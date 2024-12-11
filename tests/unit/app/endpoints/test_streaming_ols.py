"""Unit tests for streaming_ols.py."""

import pytest

from ols import config, constants
from ols.app.endpoints.streaming_ols import (
    build_yield_item,
    generic_llm_error,
    invalid_response_generator,
    prompt_too_long_error,
    yield_references,
)
from ols.app.models.models import RagChunk


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
    assert (
        build_yield_item("bla", 1, constants.MEDIA_TYPE_JSON)
        == '{"event": "token", "data": {"id": 1, "token": "bla"}}'
    )


def test_prompt_too_long_error():
    """Test prompt_too_long_error."""
    assert (
        prompt_too_long_error("error", constants.MEDIA_TYPE_TEXT)
        == "Prompt is too long: error"
    )

    assert (
        prompt_too_long_error("error", constants.MEDIA_TYPE_JSON)
        == '{"event": "error", "data": {"response": "Prompt is too long", "cause": "error"}}'
    )


def test_generic_llm_error():
    """Test generic_llm_error."""
    assert (
        generic_llm_error("error", constants.MEDIA_TYPE_TEXT)
        == "Oops, something went wrong during LLM invocation: error"
    )

    assert (
        generic_llm_error("error", constants.MEDIA_TYPE_JSON)
        == '{"event": "error", "data": {"response": "Oops, something went wrong during LLM invocation", "cause": "error"}}'  # noqa: E501
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("_load_config")
class TestYieldReferencesText:
    """Test yield_references."""

    rag_chunk_1 = RagChunk(text="bla", doc_title="title_1", doc_url="doc_url_1")
    rag_chunk_2 = RagChunk(text="bla", doc_title="title_2", doc_url="doc_url_2")
    rag_chunks = (rag_chunk_1, rag_chunk_2)

    async def test_yield_references_text(self):
        """Test yield_references in text form."""
        # without rag chunks - no odcs
        generator = yield_references([], constants.MEDIA_TYPE_TEXT)
        result = await drain_generator(generator)
        assert result == ""

        # with rag chunks
        generator = yield_references(self.rag_chunks, constants.MEDIA_TYPE_TEXT)
        result = await drain_generator(generator)
        assert result == "\n\n---\n\ntitle_1: doc_url_1\ntitle_2: doc_url_2"

    async def test_yield_references_json(self):
        """Test yield_references in json form."""
        # without rag chunks - no odcs
        generator = yield_references([], constants.MEDIA_TYPE_JSON)
        result = await drain_generator(generator)
        assert result == ""

        # with rag chunks
        generator = yield_references(self.rag_chunks, constants.MEDIA_TYPE_JSON)
        result = await drain_generator(generator)
        assert (
            result
            == '{"event": "doc_references", "data": {"doc_title": "title_1", "doc_url": "doc_url_1"}}{"event": "doc_references", "data": {"doc_title": "title_2", "doc_url": "doc_url_2"}}'  # noqa: E501
        )
