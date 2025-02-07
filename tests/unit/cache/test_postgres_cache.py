"""Unit tests for PostgresCache class."""

import json
from unittest.mock import MagicMock, call, patch

import psycopg2
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ols.app.models.config import PostgresConfig
from ols.app.models.models import CacheEntry, MessageDecoder, MessageEncoder
from ols.src.cache.cache_error import CacheError
from ols.src.cache.postgres_cache import PostgresCache
from ols.utils import suid

user_id = suid.get_suid()
conversation_id = suid.get_suid()
cache_entry_1 = CacheEntry(
    query=HumanMessage("用户消息"), response=AIMessage("人工智能信息")
)
cache_entry_2 = CacheEntry(
    query=HumanMessage("user message"), response=AIMessage("ai message")
)


@patch("psycopg2.connect")
def test_init_cache_failure_detection(mock_connect):
    """Test the exception handling for Cache.initialize_cache operation."""
    exception_message = "Exception during initializing the cache."
    mock_connect.return_value.cursor.return_value.execute.side_effect = Exception(
        exception_message
    )

    # try to connect to mocked Postgres
    config = PostgresConfig()
    with pytest.raises(Exception, match=exception_message):
        PostgresCache(config)

    # connection must be closed in case of exception
    mock_connect.return_value.close.assert_called_once_with()


@patch("psycopg2.connect")
def test_get_operation_on_empty_cache(mock_connect):
    """Test the Cache.get operation on empty cache."""
    # mock the query result - empty cache
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None
    mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

    # initialize Postgres cache
    config = PostgresConfig()
    cache = PostgresCache(config)

    # call the "get" operation
    conversation = cache.get(user_id, conversation_id)
    assert conversation == []
    mock_cursor.execute.assert_called_once_with(
        PostgresCache.SELECT_CONVERSATION_HISTORY_STATEMENT, (user_id, conversation_id)
    )
    mock_cursor.fetchone.assert_called_once()


@patch("psycopg2.connect")
def test_get_operation_invalid_value(mock_connect):
    """Test the Cache.get operation when invalid value is returned from cache."""
    # mock the query result
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = "Invalid value"
    mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

    # initialize Postgres cache
    config = PostgresConfig()
    cache = PostgresCache(config)

    # call the "get" operation
    with pytest.raises(ValueError, match="Invalid value read from cache:"):
        cache.get(user_id, conversation_id)

    # DB operation SELECT must be performed
    mock_cursor.execute.assert_called_once_with(
        PostgresCache.SELECT_CONVERSATION_HISTORY_STATEMENT, (user_id, conversation_id)
    )
    mock_cursor.fetchone.assert_called_once()


@patch("psycopg2.connect")
def test_get_operation_valid_value(mock_connect):
    """Test the Cache.get operation when valid value is returned from cache."""
    history = [
        cache_entry_1,
        cache_entry_2,
    ]
    conversation = json.dumps([ce.to_dict() for ce in history], cls=MessageEncoder)

    # mock the query result
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (conversation,)
    mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

    # initialize Postgres cache
    config = PostgresConfig()
    cache = PostgresCache(config)

    # call the "get" operation
    # unjsond history should be returned
    assert cache.get(user_id, conversation_id) == history

    # DB operation SELECT must be performed
    mock_cursor.execute.assert_called_once_with(
        PostgresCache.SELECT_CONVERSATION_HISTORY_STATEMENT, (user_id, conversation_id)
    )
    mock_cursor.fetchone.assert_called_once()


@patch("psycopg2.connect")
def test_get_operation_on_exception(mock_connect):
    """Test the Cache.get operation when exception is thrown."""
    # initialize Postgres cache
    config = PostgresConfig()
    cache = PostgresCache(config)

    # mock the query
    mock_cursor = MagicMock()
    mock_cursor.fetchone.side_effect = psycopg2.DatabaseError("PLSQL error")
    mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

    # error must be raised during cache operation
    with pytest.raises(CacheError, match="PLSQL error"):
        cache.get(user_id, conversation_id)


@patch("psycopg2.connect")
def test_insert_or_append_operation(mock_connect):
    """Test the Cache.insert_or_append operation for first item to be inserted."""
    history = cache_entry_1
    conversation = json.dumps([history.to_dict()], cls=MessageEncoder)

    # mock the query result
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None
    mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

    # initialize Postgres cache
    config = PostgresConfig()
    cache = PostgresCache(config)

    # call the "insert_or_append" operation
    # to insert new conversation history
    cache.insert_or_append(user_id, conversation_id, history)

    # multiple DB operations must be performed
    calls = [
        call(
            PostgresCache.SELECT_CONVERSATION_HISTORY_STATEMENT,
            (user_id, conversation_id),
        ),
        call(
            PostgresCache.INSERT_CONVERSATION_HISTORY_STATEMENT,
            (user_id, conversation_id, conversation.encode("utf-8")),
        ),
        call(PostgresCache.QUERY_CACHE_SIZE),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=True)


@patch("psycopg2.connect")
def test_insert_or_append_operation_append_item(mock_connect):
    """Test the Cache.insert_or_append operation for more item to be inserted."""
    stored_history = cache_entry_1

    old_conversation = json.dumps([stored_history.to_dict()], cls=MessageEncoder)

    appended_history = cache_entry_2

    # create json object in the exactly same format
    whole_history = json.loads(old_conversation, cls=MessageDecoder)
    whole_history.append(appended_history.to_dict())
    new_conversation = json.dumps(whole_history, cls=MessageEncoder)

    # mock the query result
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (old_conversation,)
    mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

    # initialize Postgres cache
    config = PostgresConfig()
    cache = PostgresCache(config)

    # call the "insert_or_append" operation
    # to append new history to the old one
    cache.insert_or_append(user_id, conversation_id, appended_history)

    # multiple DB operations must be performed
    calls = [
        call(
            PostgresCache.SELECT_CONVERSATION_HISTORY_STATEMENT,
            (user_id, conversation_id),
        ),
        call(
            PostgresCache.UPDATE_CONVERSATION_HISTORY_STATEMENT,
            (new_conversation.encode("utf-8"), user_id, conversation_id),
        ),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=True)


@patch("psycopg2.connect")
def test_insert_or_append_operation_on_exception(mock_connect):
    """Test the Cache.insert_or_append operation when exception is thrown."""
    history = cache_entry_1

    # mock the query result
    mock_cursor = MagicMock()
    mock_cursor.fetchone.side_effect = psycopg2.DatabaseError("PLSQL error")
    mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

    # initialize Postgres cache
    config = PostgresConfig()
    cache = PostgresCache(config)

    # error must be raised during cache operation
    with pytest.raises(CacheError, match="PLSQL error"):
        cache.insert_or_append(user_id, conversation_id, history)


@patch("psycopg2.connect")
def test_list_operation(mock_connect):
    """Test the Cache.list operation."""
    # Mock conversation IDs to be returned by the database
    mock_conversation_ids = ["conversation_1", "conversation_2", "conversation_3"]

    # Mock the database cursor behavior
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [(cid,) for cid in mock_conversation_ids]
    mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

    # Initialize Postgres cache
    config = PostgresConfig()
    cache = PostgresCache(config)

    # Call the "list" operation
    conversation_ids = cache.list(user_id)

    # Verify the result
    assert conversation_ids == mock_conversation_ids

    # Verify the query execution
    mock_cursor.execute.assert_called_once_with(
        PostgresCache.LIST_CONVERSATIONS_STATEMENT, (user_id,)
    )
    mock_cursor.fetchall.assert_called_once()


@patch("psycopg2.connect")
def test_list_operation_on_exception(mock_connect):
    """Test the Cache.list operation when an exception is raised."""
    # Mock the database cursor behavior to raise an exception
    mock_cursor = MagicMock()
    mock_cursor.fetchall.side_effect = psycopg2.DatabaseError("PLSQL error")
    mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

    # Initialize Postgres cache
    config = PostgresConfig()
    cache = PostgresCache(config)

    # Verify that the exception is raised
    with pytest.raises(CacheError, match="PLSQL error"):
        cache.list(user_id)


@patch("psycopg2.connect")
def test_delete_operation(mock_connect):
    """Test the Cache.delete operation."""
    # Mock the database cursor behavior
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = True
    mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

    # Initialize Postgres cache
    config = PostgresConfig()
    cache = PostgresCache(config)

    # Call the "delete" operation
    result = cache.delete(user_id, conversation_id)

    # Verify the result
    assert result is True

    # Verify the query execution
    mock_cursor.execute.assert_called_once_with(
        PostgresCache.DELETE_SINGLE_CONVERSATION_STATEMENT, (user_id, conversation_id)
    )
    mock_cursor.fetchone.assert_called_once()


@patch("psycopg2.connect")
def test_delete_operation_not_found(mock_connect):
    """Test the Cache.delete operation when the conversation is not found."""
    # Mock the database cursor behavior to simulate no row found
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None
    mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

    # Initialize Postgres cache
    config = PostgresConfig()
    cache = PostgresCache(config)

    # Call the "delete" operation
    result = cache.delete(user_id, conversation_id)

    # Verify the result
    assert result is False

    # Verify the query execution
    mock_cursor.execute.assert_called_once_with(
        PostgresCache.DELETE_SINGLE_CONVERSATION_STATEMENT, (user_id, conversation_id)
    )
    mock_cursor.fetchone.assert_called_once()


@patch("psycopg2.connect")
def test_delete_operation_on_exception(mock_connect):
    """Test the Cache.delete operation when an exception is raised."""
    # Mock the database cursor behavior to raise an exception
    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = psycopg2.DatabaseError("PLSQL error")
    mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

    # Initialize Postgres cache
    config = PostgresConfig()
    cache = PostgresCache(config)

    # Verify that the exception is raised
    with pytest.raises(CacheError, match="PLSQL error"):
        cache.delete(user_id, conversation_id)
