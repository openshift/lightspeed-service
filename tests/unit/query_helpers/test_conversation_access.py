"""Unit tests for conversation access enforcement."""

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage

from ols import config
from ols.app.models.config import InMemoryCacheConfig
from ols.app.models.models import CacheEntry
from ols.constants import DEFAULT_USER_UID
from ols.src.cache.in_memory_cache import InMemoryCache
from ols.src.query_helpers.conversation_access import ensure_conversation_owned_by_user
from ols.utils import suid

cache_entry = CacheEntry(
    query=HumanMessage("user message"), response=AIMessage("ai message")
)


@pytest.fixture(name="memory_cache")
def fixture_memory_cache(monkeypatch):
    """In-memory cache for ownership tests."""
    cache = InMemoryCache(InMemoryCacheConfig({"max_entries": "32"}))
    monkeypatch.setattr(config, "_conversation_cache", cache)
    return cache


def test_allows_requester_owned_conversation(memory_cache):
    """No error when conversation_id is owned by the requester."""
    conversation_id = suid.get_suid()
    owner = DEFAULT_USER_UID
    memory_cache.insert_or_append(owner, conversation_id, cache_entry, False)

    ensure_conversation_owned_by_user(owner, conversation_id)


def test_blocks_other_users_conversation(memory_cache):
    """Reject when conversation_id is bound to a different user_id."""
    conversation_id = suid.get_suid()
    owner = DEFAULT_USER_UID
    other = suid.get_suid()
    memory_cache.insert_or_append(owner, conversation_id, cache_entry, False)

    with pytest.raises(HTTPException) as exc_info:
        ensure_conversation_owned_by_user(other, conversation_id)

    assert exc_info.value.status_code == 403


def test_allows_unknown_conversation_id(memory_cache):
    """New conversation_id values are allowed before first cache write."""
    ensure_conversation_owned_by_user(suid.get_suid(), suid.get_suid())