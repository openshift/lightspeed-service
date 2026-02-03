"""Unit tests for history support helpers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ols import config
from ols.app.models.models import CacheEntry
from ols.src.query_helpers.history_support import (
    compress_conversation_history,
    summarize_entries,
)
from ols.utils import suid


@pytest.fixture(scope="function", autouse=True)
def _setup():
    """Set up config for tests."""
    config.reload_from_yaml_file("tests/config/valid_config_without_mcp.yaml")


@pytest.mark.asyncio
async def test_summarize_entries_success():
    """Test summarize_entries with successful LLM summarization."""
    entries = [
        CacheEntry(
            query=HumanMessage(content="What is Kubernetes?"),
            response=AIMessage(
                content="Kubernetes is a container orchestration platform."
            ),
        ),
        CacheEntry(
            query=HumanMessage(content="How do I create a pod?"),
            response=AIMessage(content="Use kubectl create pod command."),
        ),
    ]
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=AIMessage(
            content="Summary: Discussion about Kubernetes basics and pod creation."
        )
    )

    summary = await summarize_entries(entries, mock_llm)

    assert summary == "Summary: Discussion about Kubernetes basics and pod creation."
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_summarize_entries_empty():
    """Test summarize_entries with empty entries list."""
    summary = await summarize_entries([], MagicMock())
    assert summary is None


@pytest.mark.asyncio
async def test_summarize_entries_llm_failure(caplog):
    """Test summarize_entries when LLM fails."""
    entries = [
        CacheEntry(
            query=HumanMessage(content="Test query"),
            response=AIMessage(content="Test response"),
        ),
    ]
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

    summary = await summarize_entries(entries, mock_llm)

    assert summary is None
    assert "Failed to summarize conversation entries" in caplog.text


@pytest.mark.asyncio
async def test_summarize_entries_with_retry_on_transient_error(caplog):
    """Test summarize_entries retries on transient errors."""
    entries = [
        CacheEntry(
            query=HumanMessage(content="What is Kubernetes?"),
            response=AIMessage(
                content="Kubernetes is a container orchestration platform."
            ),
        ),
    ]
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[Exception("Connection timeout"), AIMessage(content="Summary text")]
    )

    with patch(
        "ols.src.query_helpers.history_support.asyncio.sleep", new_callable=AsyncMock
    ):
        summary = await summarize_entries(entries, mock_llm)

    assert summary == "Summary text"
    assert mock_llm.ainvoke.call_count == 2
    assert "Transient error on attempt 1/3" in caplog.text


@pytest.mark.asyncio
async def test_summarize_entries_with_retry_exhausted(caplog):
    """Test summarize_entries gives up after max retries on transient errors."""
    entries = [
        CacheEntry(
            query=HumanMessage(content="Test query"),
            response=AIMessage(content="Test response"),
        ),
    ]
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("Rate limit exceeded"))

    with patch(
        "ols.src.query_helpers.history_support.asyncio.sleep", new_callable=AsyncMock
    ):
        summary = await summarize_entries(entries, mock_llm)

    assert summary is None
    assert mock_llm.ainvoke.call_count == 3
    assert (
        "Failed to summarize conversation entries: Rate limit exceeded" in caplog.text
    )


@pytest.mark.asyncio
async def test_summarize_entries_no_retry_on_permanent_error(caplog):
    """Test summarize_entries does not retry on permanent errors."""
    entries = [
        CacheEntry(
            query=HumanMessage(content="Test query"),
            response=AIMessage(content="Test response"),
        ),
    ]
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("Authentication failed"))

    summary = await summarize_entries(entries, mock_llm)

    assert summary is None
    assert mock_llm.ainvoke.call_count == 1
    assert (
        "Failed to summarize conversation entries: Authentication failed" in caplog.text
    )
    assert "Transient error" not in caplog.text


@pytest.mark.asyncio
async def test_compress_conversation_history_no_compression_needed():
    """Test oldest entry is summarized when keep threshold is met."""
    conversation_id = suid.get_suid()
    user_id = "test_user"
    cache_entries = [
        CacheEntry(
            query=HumanMessage(content=f"Query {i}"),
            response=AIMessage(content=f"Response {i}"),
        )
        for i in range(5)
    ]

    with (
        patch("ols.config.conversation_cache.delete") as mock_cache_delete,
        patch("ols.config.conversation_cache.insert_or_append") as mock_cache_insert,
    ):
        result = await compress_conversation_history(
            user_id,
            conversation_id,
            True,
            provider="p",
            model="m",
            bare_llm=MagicMock(
                ainvoke=AsyncMock(return_value=AIMessage(content="summary"))
            ),
            full_cache_entries=cache_entries,
            kept_newest_first=list(reversed(cache_entries)),
        )

    assert len(result) == 5
    assert result[0].query.content == "[Previous conversation summary]"
    assert result[1:] == cache_entries[:-1]
    mock_cache_delete.assert_called_once_with(user_id, conversation_id, True)
    assert mock_cache_insert.call_count == 5


@pytest.mark.asyncio
async def test_compress_conversation_history_successful_compression():
    """Test compression path with successful summary creation."""
    conversation_id = suid.get_suid()
    user_id = "test_user"
    cache_entries = [
        CacheEntry(
            query=HumanMessage(content=f"Query {i}"),
            response=AIMessage(content=f"Response {i}"),
        )
        for i in range(10)
    ]

    with (
        patch(
            "ols.src.query_helpers.history_support.summarize_entries",
            new=AsyncMock(return_value="Summary of first 5 conversations"),
        ),
        patch("ols.config.conversation_cache.delete") as mock_cache_delete,
        patch("ols.config.conversation_cache.insert_or_append") as mock_cache_insert,
    ):
        result = await compress_conversation_history(
            user_id,
            conversation_id,
            True,
            provider="p",
            model="m",
            bare_llm=MagicMock(),
            full_cache_entries=cache_entries,
            kept_newest_first=list(reversed(cache_entries[-6:])),
        )

    assert len(result) == 6
    assert result[0].query.content == "[Previous conversation summary]"
    assert result[0].response.content == "Summary of first 5 conversations"
    mock_cache_delete.assert_called_once_with(user_id, conversation_id, True)
    assert mock_cache_insert.call_count == 6


@pytest.mark.asyncio
async def test_compress_conversation_history_summarization_failure(caplog):
    """Test compression falls back to keep_entries on summary failure."""
    conversation_id = suid.get_suid()
    user_id = "test_user"
    cache_entries = [
        CacheEntry(
            query=HumanMessage(content=f"Query {i}"),
            response=AIMessage(content=f"Response {i}"),
        )
        for i in range(10)
    ]

    with (
        patch(
            "ols.src.query_helpers.history_support.summarize_entries",
            new=AsyncMock(return_value=None),
        ),
        patch("ols.config.conversation_cache.delete") as mock_cache_delete,
        patch("ols.config.conversation_cache.insert_or_append") as mock_cache_insert,
    ):
        result = await compress_conversation_history(
            user_id,
            conversation_id,
            True,
            provider="p",
            model="m",
            bare_llm=MagicMock(),
            full_cache_entries=cache_entries,
            kept_newest_first=list(reversed(cache_entries[-6:])),
        )

    assert len(result) == 5
    assert result == cache_entries[-5:]
    mock_cache_delete.assert_called_once_with(user_id, conversation_id, True)
    assert mock_cache_insert.call_count == 5
    assert "Summarization failed" in caplog.text


@pytest.mark.asyncio
async def test_compress_conversation_history_cache_update_failure(caplog):
    """Test compression falls back when cache update fails."""
    conversation_id = suid.get_suid()
    user_id = "test_user"
    cache_entries = [
        CacheEntry(
            query=HumanMessage(content=f"Query {i}"),
            response=AIMessage(content=f"Response {i}"),
        )
        for i in range(10)
    ]

    with (
        patch(
            "ols.src.query_helpers.history_support.summarize_entries",
            new=AsyncMock(return_value="Summary of conversations"),
        ),
        patch(
            "ols.config.conversation_cache.delete", side_effect=Exception("Cache error")
        ),
    ):
        result = await compress_conversation_history(
            user_id,
            conversation_id,
            True,
            provider="p",
            model="m",
            bare_llm=MagicMock(),
            full_cache_entries=cache_entries,
            kept_newest_first=list(reversed(cache_entries[-6:])),
        )

    assert result == []
    assert "Failed to update cache with compressed history" in caplog.text


@pytest.mark.asyncio
async def test_compress_conversation_history_compresses_small_count_when_token_limited():
    """Test compression when entries are few but exceed token budget."""
    conversation_id = suid.get_suid()
    user_id = "test_user"
    cache_entries = [
        CacheEntry(
            query=HumanMessage(content=f"Long query {i}"),
            response=AIMessage(content=f"Long response {i}"),
        )
        for i in range(3)
    ]

    with (
        patch(
            "ols.src.query_helpers.history_support.summarize_entries",
            new=AsyncMock(return_value="summary"),
        ),
        patch("ols.config.conversation_cache.delete") as mock_cache_delete,
        patch("ols.config.conversation_cache.insert_or_append") as mock_cache_insert,
    ):
        result = await compress_conversation_history(
            user_id,
            conversation_id,
            True,
            provider="p",
            model="m",
            bare_llm=MagicMock(),
            full_cache_entries=cache_entries,
            kept_newest_first=list(reversed(cache_entries)),
            entries_to_keep=2,
        )

    assert len(result) == 3
    assert result[0].query.content == "[Previous conversation summary]"
    assert result[0].response.content == "summary"
    mock_cache_delete.assert_called_once_with(user_id, conversation_id, True)
    assert mock_cache_insert.call_count == 3
