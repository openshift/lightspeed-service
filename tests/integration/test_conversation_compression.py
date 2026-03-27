"""Integration tests for conversation compression/summarization."""

# we add new attributes into pytest instance, which is not recognized
# properly by linters
# pyright: reportAttributeAccessIssue=false

from unittest.mock import patch

import pytest
import requests
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from ols import config
from ols.app.models.models import CacheEntry
from ols.constants import DEFAULT_USER_UID
from ols.utils import suid
from tests.mock_classes.mock_langchain_interface import mock_langchain_interface
from tests.mock_classes.mock_llm_loader import mock_llm_loader


@pytest.fixture(scope="function")
def _setup():
    """Set up the test client."""
    config.reload_from_yaml_file("tests/config/config_for_compression_tests.yaml")

    from ols.app.main import app  # pylint: disable=import-outside-toplevel

    pytest.client = TestClient(app)  # type: ignore[attr-defined]


def test_conversation_compression_triggered(_setup: None) -> None:
    """Verify that long conversations trigger automatic compression.

    This test:
    1. Populates cache with 30 entries
    2. Makes a query that triggers compression
    3. Verifies compression happened (30 → 7 entries: 1 summary + 5 recent + 1 new)
    4. Verifies summary structure and content
    """
    ml = mock_langchain_interface(
        AIMessage(content="Summary of conversation about features and topics")
    )

    with patch("ols.src.query_helpers.query_helper.load_llm", new=mock_llm_loader(ml)):
        user_id = DEFAULT_USER_UID
        cid = suid.get_suid()

        # Populate cache with 30 entries
        for i in range(30):
            entry = CacheEntry(
                query=HumanMessage(content=f"Question {i}: What is feature-{i}?"),
                response=AIMessage(content=f"Feature-{i} is about topic-{i} details."),
            )
            config.conversation_cache.insert_or_append(
                user_id, cid, entry, skip_user_id_check=True
            )

        # Verify cache has 25 entries before compression
        cached_entries = config.conversation_cache.get(
            user_id, cid, skip_user_id_check=True
        )
        assert len(cached_entries) == 30, "Cache should have 30 entries before request"

        # Make request that triggers compression
        response = pytest.client.post(  # type: ignore[attr-defined]
            "/v1/query",
            json={"conversation_id": cid, "query": "What is a pod?"},
        )

        assert response.status_code == requests.codes.ok
        json_response = response.json()
        assert len(json_response["response"]) > 0

        # Verify compression happened
        cached_entries_after = config.conversation_cache.get(
            user_id, cid, skip_user_id_check=True
        )
        assert len(cached_entries_after) == 7, (
            f"Cache should be compressed to 7 entries "
            f"(1 summary + 5 recent + 1 new), got {len(cached_entries_after)}"
        )

        # Verify first entry is the summary with the marker
        first_query_content = str(cached_entries_after[0].query.content)
        assert (
            "[Previous conversation summary]" in first_query_content
        ), f"First entry should be summary, got: {first_query_content}"

        # Verify the summary response contains content
        summary_response = str(cached_entries_after[0].response.content)
        assert len(summary_response) > 0, "Summary should have content"


def test_no_recompression_after_compression(_setup: None) -> None:
    """Verify that compression doesn't happen again immediately after compression.

    This test:
    1. Populates cache with 30 entries and triggers compression
    2. Makes 2 more queries
    3. Verifies NO new compression happened (summary stays the same)
    4. Verifies entries are just added (7 → 8 → 9)
    """
    ml = mock_langchain_interface(
        AIMessage(content="Summary of conversation about features")
    )

    with patch("ols.src.query_helpers.query_helper.load_llm", new=mock_llm_loader(ml)):
        user_id = DEFAULT_USER_UID
        cid = suid.get_suid()

        # Populate cache with 30 entries
        for i in range(30):
            entry = CacheEntry(
                query=HumanMessage(content=f"Question {i}: What is feature-{i}?"),
                response=AIMessage(content=f"Feature-{i} is about topic-{i} details."),
            )
            config.conversation_cache.insert_or_append(
                user_id, cid, entry, skip_user_id_check=True
            )

        # Trigger compression
        response1 = pytest.client.post(  # type: ignore[attr-defined]
            "/v1/query",
            json={"conversation_id": cid, "query": "What is a pod?"},
        )
        assert response1.status_code == requests.codes.ok

        # Get the summary after first compression
        cached_entries = config.conversation_cache.get(
            user_id, cid, skip_user_id_check=True
        )
        assert len(cached_entries) == 7
        summary_content = cached_entries[0].response.content
        summary_query = cached_entries[0].query.content

        # Make another query
        response2 = pytest.client.post(  # type: ignore[attr-defined]
            "/v1/query",
            json={"conversation_id": cid, "query": "What is a namespace?"},
        )
        assert response2.status_code == requests.codes.ok

        # Verify: Cache grew by 1, summary unchanged
        cached_entries = config.conversation_cache.get(
            user_id, cid, skip_user_id_check=True
        )
        assert (
            len(cached_entries) == 8
        ), f"Cache should grow to 8 entries (no recompression), got {len(cached_entries)}"
        assert (
            cached_entries[0].response.content == summary_content
        ), "Summary should not change"
        assert (
            cached_entries[0].query.content == summary_query
        ), "Summary query should not change"

        # Make one more query
        response3 = pytest.client.post(  # type: ignore[attr-defined]
            "/v1/query",
            json={"conversation_id": cid, "query": "What is a deployment?"},
        )
        assert response3.status_code == requests.codes.ok

        # Verify: Cache grew by 1 again, summary still unchanged
        cached_entries = config.conversation_cache.get(
            user_id, cid, skip_user_id_check=True
        )
        assert len(cached_entries) == 9, (
            f"Cache should grow to 9 entries (still no recompression), "
            f"got {len(cached_entries)}"
        )
        assert (
            cached_entries[0].response.content == summary_content
        ), "Summary should still not change"
        assert (
            cached_entries[0].query.content == summary_query
        ), "Summary query should still not change"


def test_compression_triggers_again_at_threshold(_setup: None) -> None:
    """Verify that compression triggers again when threshold is reached again.

    This test:
    1. Populates cache with 30 entries and triggers compression (→ 7 entries)
    2. Adds 23 more entries to reach 30 again
    3. Makes a query to trigger second compression
    4. Verifies compression happened again with a NEW summary
    """
    # First compression returns first summary
    ml1 = mock_langchain_interface(AIMessage(content="First summary of conversation"))

    with patch("ols.src.query_helpers.query_helper.load_llm", new=mock_llm_loader(ml1)):
        user_id = DEFAULT_USER_UID
        cid = suid.get_suid()

        # Populate cache with 30 entries
        for i in range(30):
            entry = CacheEntry(
                query=HumanMessage(content=f"Question {i}: What is feature-{i}?"),
                response=AIMessage(content=f"Feature-{i} is about topic-{i} details."),
            )
            config.conversation_cache.insert_or_append(
                user_id, cid, entry, skip_user_id_check=True
            )

        # Trigger first compression
        response1 = pytest.client.post(  # type: ignore[attr-defined]
            "/v1/query",
            json={"conversation_id": cid, "query": "Query 1"},
        )
        assert response1.status_code == requests.codes.ok

        cached_entries = config.conversation_cache.get(
            user_id, cid, skip_user_id_check=True
        )
        assert len(cached_entries) == 7
        first_summary = cached_entries[0].response.content

    # Add 18 more entries to reach 30 again (7 + 23 = 30)
    for i in range(25, 48):
        entry = CacheEntry(
            query=HumanMessage(content=f"Question {i}: What is feature-{i}?"),
            response=AIMessage(content=f"Feature-{i} is about topic-{i} details."),
        )
        config.conversation_cache.insert_or_append(
            user_id, cid, entry, skip_user_id_check=True
        )

    # Verify we're back at 30
    cached_entries = config.conversation_cache.get(
        user_id, cid, skip_user_id_check=True
    )
    assert len(cached_entries) == 30

    # Second compression returns different summary
    ml2 = mock_langchain_interface(
        AIMessage(content="Second summary with more recent context")
    )

    with patch("ols.src.query_helpers.query_helper.load_llm", new=mock_llm_loader(ml2)):
        # Trigger second compression
        response2 = pytest.client.post(  # type: ignore[attr-defined]
            "/v1/query",
            json={"conversation_id": cid, "query": "Query 2"},
        )
        assert response2.status_code == requests.codes.ok

        # Verify compression happened again
        cached_entries = config.conversation_cache.get(
            user_id, cid, skip_user_id_check=True
        )
        assert (
            len(cached_entries) == 7
        ), f"Cache should be compressed to 7 entries again, got {len(cached_entries)}"

        # Verify NEW summary was created
        second_summary = cached_entries[0].response.content
        assert (
            second_summary != first_summary
        ), "Second compression should create a new summary"
        assert "Second summary" in str(
            second_summary
        ), f"Should have new summary content, got: {second_summary}"
