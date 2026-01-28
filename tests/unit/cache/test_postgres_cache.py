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


def test_init_cache_failure_detection():
    """Test the exception handling for Cache.initialize_cache operation."""
    exception_message = "Exception during initializing the cache."

    # do not use real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.execute.side_effect = Exception(
            exception_message
        )

        # try to connect to mocked Postgres
        config = PostgresConfig()
        with pytest.raises(Exception, match=exception_message):
            PostgresCache(config)

        # connection must be closed in case of exception
        mock_connect.return_value.close.assert_called_once_with()


def test_get_operation_on_empty_cache():
    """Test the Cache.get operation on empty cache."""
    # mock the query result - empty cache
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None

    # do not use real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # initialize Postgres cache
        config = PostgresConfig()
        cache = PostgresCache(config)

    # call the "get" operation
    conversation = cache.get(user_id, conversation_id)
    assert conversation == []

    # multiple DB operations must be performed:
    # 1. check if connection to DB is alive
    # 2. select conversation from DB
    calls = [
        call("SELECT 1"),
        call(
            PostgresCache.SELECT_CONVERSATION_HISTORY_STATEMENT,
            (user_id, conversation_id),
        ),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=False)

    # Verify the query execution
    mock_cursor.fetchone.assert_called_once()


def test_get_operation_invalid_value():
    """Test the Cache.get operation when invalid value is returned from cache."""
    # mock the query result
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = "Invalid value"

    # do not use real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # initialize Postgres cache
        config = PostgresConfig()
        cache = PostgresCache(config)

        # call the "get" operation
        with pytest.raises(ValueError, match="Invalid value read from cache:"):
            cache.get(user_id, conversation_id)

    # multiple DB operations must be performed:
    # 1. check if connection to DB is alive
    # 2. select conversation from DB
    calls = [
        call("SELECT 1"),
        call(
            PostgresCache.SELECT_CONVERSATION_HISTORY_STATEMENT,
            (user_id, conversation_id),
        ),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=False)

    # Verify the query execution
    mock_cursor.fetchone.assert_called_once()


def test_get_operation_valid_value():
    """Test the Cache.get operation when valid value is returned from cache."""
    history = [
        cache_entry_1,
        cache_entry_2,
    ]
    conversation = json.dumps([ce.to_dict() for ce in history], cls=MessageEncoder)
    as_memview = memoryview(bytearray(conversation, "utf-8"))

    # mock the query result
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (as_memview,)

    # do not use real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # initialize Postgres cache
        config = PostgresConfig()
        cache = PostgresCache(config)

    # call the "get" operation
    # unjsond history should be returned
    assert cache.get(user_id, conversation_id) == history

    # multiple DB operations must be performed:
    # 1. check if connection to DB is alive
    # 2. select conversation from DB
    calls = [
        call("SELECT 1"),
        call(
            PostgresCache.SELECT_CONVERSATION_HISTORY_STATEMENT,
            (user_id, conversation_id),
        ),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=False)

    # Verify the query execution
    mock_cursor.fetchone.assert_called_once()


def test_get_operation_on_exception():
    """Test the Cache.get operation when exception is thrown."""
    # mock the query
    mock_cursor = MagicMock()
    mock_cursor.fetchone.side_effect = psycopg2.DatabaseError("PLSQL error")

    # do not use real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # initialize Postgres cache
        config = PostgresConfig()
        cache = PostgresCache(config)

    # error must be raised during cache operation
    with pytest.raises(CacheError, match="PLSQL error"):
        cache.get(user_id, conversation_id)


def test_get_operation_on_disconnected_db():
    """Test the Cache.get operation when DB is not connected."""
    # mock the query
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None

    # do not use real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # initialize Postgres cache
        config = PostgresConfig()
        cache = PostgresCache(config)
        # simulate DB disconnection
        cache.connection = None
        assert not cache.connected()
        # DB operation should connect automatically
        cache.get(user_id, conversation_id)
        assert cache.connected()

    calls = [
        call(
            PostgresCache.SELECT_CONVERSATION_HISTORY_STATEMENT,
            (user_id, conversation_id),
        ),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=False)


def test_insert_or_append_operation():
    """Test the Cache.insert_or_append operation for first item to be inserted."""
    history = cache_entry_1
    conversation = json.dumps([history.to_dict()], cls=MessageEncoder)

    # mock the query result
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None

    # do not use real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # initialize Postgres cache
        config = PostgresConfig()
        cache = PostgresCache(config)

        # call the "insert_or_append" operation
        # to insert new conversation history
        cache.insert_or_append(user_id, conversation_id, history)

    # multiple DB operations must be performed:
    calls = [
        call(
            PostgresCache.SELECT_CONVERSATION_HISTORY_STATEMENT,
            (user_id, conversation_id),
        ),
        call(
            PostgresCache.INSERT_CONVERSATION_HISTORY_STATEMENT,
            (user_id, conversation_id, conversation.encode("utf-8")),
        ),
        call(
            PostgresCache.UPSERT_CONVERSATION_STATEMENT,
            (user_id, conversation_id),
        ),
        call(PostgresCache.QUERY_TOTAL_ENTRIES),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=False)


def test_insert_or_append_operation_append_item():
    """Test the Cache.insert_or_append operation for more item to be inserted."""
    stored_history = cache_entry_1

    old_conversation = json.dumps([stored_history.to_dict()], cls=MessageEncoder)
    as_memview = memoryview(bytearray(old_conversation, "utf-8"))

    appended_history = cache_entry_2

    # create json object in the exactly same format
    whole_history = json.loads(old_conversation, cls=MessageDecoder)
    whole_history.append(appended_history.to_dict())
    new_conversation = json.dumps(whole_history, cls=MessageEncoder)

    # mock the query result
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (as_memview,)

    # do not use real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # initialize Postgres cache
        config = PostgresConfig()
        cache = PostgresCache(config)

        # call the "insert_or_append" operation
        # to append new history to the old one
        cache.insert_or_append(user_id, conversation_id, appended_history)

    # multiple DB operations must be performed:
    calls = [
        call(
            PostgresCache.SELECT_CONVERSATION_HISTORY_STATEMENT,
            (user_id, conversation_id),
        ),
        call(
            PostgresCache.UPDATE_CONVERSATION_HISTORY_STATEMENT,
            (new_conversation.encode("utf-8"), user_id, conversation_id),
        ),
        call(
            PostgresCache.UPSERT_CONVERSATION_STATEMENT,
            (user_id, conversation_id),
        ),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=False)


def test_insert_or_append_operation_on_exception():
    """Test the Cache.insert_or_append operation when exception is thrown."""
    history = cache_entry_1

    # mock the query result
    mock_cursor = MagicMock()
    mock_cursor.fetchone.side_effect = psycopg2.DatabaseError("PLSQL error")

    # do not use real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # initialize Postgres cache
        config = PostgresConfig()
        cache = PostgresCache(config)

        # error must be raised during cache operation
        with pytest.raises(CacheError, match="PLSQL error"):
            cache.insert_or_append(user_id, conversation_id, history)


def test_insert_or_append_operation_on_disconnected_db():
    """Test the Cache.insert_or_append operation when DB is not connected."""
    history = cache_entry_1
    conversation = json.dumps([history.to_dict()], cls=MessageEncoder)

    # mock the query
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None

    # do not use real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # initialize Postgres cache
        config = PostgresConfig()
        cache = PostgresCache(config)
        # simulate DB disconnection
        cache.connection = None
        assert not cache.connected()
        # DB operation should connect automatically
        cache.insert_or_append(user_id, conversation_id, cache_entry_1)
        assert cache.connected()

    # multiple DB operations must be performed:
    calls = [
        call(
            PostgresCache.SELECT_CONVERSATION_HISTORY_STATEMENT,
            (user_id, conversation_id),
        ),
        call(
            PostgresCache.INSERT_CONVERSATION_HISTORY_STATEMENT,
            (user_id, conversation_id, conversation.encode("utf-8")),
        ),
        call(
            PostgresCache.UPSERT_CONVERSATION_STATEMENT,
            (user_id, conversation_id),
        ),
        call(PostgresCache.QUERY_TOTAL_ENTRIES),
        call("SELECT 1"),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=False)


def test_list_operation():
    """Test the Cache.list operation."""
    # Mock conversation data to be returned by the database
    # Format: (conversation_id, topic_summary, last_message_timestamp, message_count)
    mock_conversations = [
        ("conversation_1", "First topic", 1737370500.0, 2),
        ("conversation_2", "Second topic", 1737370600.0, 5),
        ("conversation_3", "Third topic", 1737370700.0, 3),
    ]

    # Mock the database cursor behavior
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = mock_conversations

    # do not use real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # Initialize Postgres cache
        config = PostgresConfig()
        cache = PostgresCache(config)

        # Call the "list" operation
        result = cache.list(user_id)

    # Verify the result matches the expected format (list of ConversationData)
    assert len(result) == 3
    assert result[0].conversation_id == "conversation_1"
    assert result[0].topic_summary == "First topic"
    assert result[0].last_message_timestamp == 1737370500.0
    assert result[0].message_count == 2
    assert result[1].conversation_id == "conversation_2"
    assert result[2].conversation_id == "conversation_3"

    # multiple DB operations must be performed:
    # 1. check if connection to DB is alive
    # 2. list conversations from DB
    calls = [
        call("SELECT 1"),
        call(PostgresCache.LIST_CONVERSATIONS_STATEMENT, (user_id,)),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=False)

    # Verify the query execution
    mock_cursor.fetchall.assert_called_once()


def test_list_operation_on_exception():
    """Test the Cache.list operation when an exception is raised."""
    # Mock the database cursor behavior to raise an exception
    mock_cursor = MagicMock()
    mock_cursor.fetchall.side_effect = psycopg2.DatabaseError("PLSQL error")

    # do not use real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # Initialize Postgres cache
        config = PostgresConfig()
        cache = PostgresCache(config)

        # Verify that the exception is raised
        with pytest.raises(CacheError, match="PLSQL error"):
            cache.list(user_id)


def test_list_operation_on_disconnected_db():
    """Test the Cache.list operation when DB is not connected."""
    # mock the query
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None
    mock_cursor.fetchall.return_value = []

    # do not use real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # initialize Postgres cache
        config = PostgresConfig()
        cache = PostgresCache(config)
        # simulate DB disconnection
        cache.connection = None
        assert not cache.connected()
        # DB operation should connect automatically
        cache.list(user_id)
        assert cache.connected()

    # one DB operation must be performed:
    # 1. list conversations from DB
    calls = [
        call(PostgresCache.LIST_CONVERSATIONS_STATEMENT, (user_id,)),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=False)


def test_set_topic_summary_operation():
    """Test the Cache.set_topic_summary operation."""
    # Mock the database cursor behavior
    mock_cursor = MagicMock()

    # do not use real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # Initialize Postgres cache
        config = PostgresConfig()
        cache = PostgresCache(config)

        # Call the "set_topic_summary" operation
        cache.set_topic_summary(user_id, conversation_id, "Test Topic Summary")

    # multiple DB operations must be performed:
    # 1. check if connection to DB is alive
    # 2. upsert topic summary
    calls = [
        call("SELECT 1"),
        call(
            PostgresCache.INSERT_OR_UPDATE_TOPIC_SUMMARY_STATEMENT,
            (user_id, conversation_id, "Test Topic Summary"),
        ),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=False)


def test_set_topic_summary_operation_on_exception():
    """Test the Cache.set_topic_summary operation when an exception is raised."""
    # Mock the database cursor behavior to raise an exception on the second execute call
    # (first call is "SELECT 1" for connection check)
    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = [
        None,  # SELECT 1 succeeds
        psycopg2.DatabaseError("PLSQL error"),  # actual operation fails
    ]

    # do not use real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # Initialize Postgres cache
        config = PostgresConfig()
        cache = PostgresCache(config)

        # Verify that the exception is raised
        with pytest.raises(CacheError, match="PLSQL error"):
            cache.set_topic_summary(user_id, conversation_id, "Test Topic")


def test_delete_operation():
    """Test the Cache.delete operation."""
    # Mock the database cursor behavior
    mock_cursor = MagicMock()
    mock_cursor.rowcount = 1

    # do not use real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # Initialize Postgres cache
        config = PostgresConfig()
        cache = PostgresCache(config)

        # Call the "delete" operation
        result = cache.delete(user_id, conversation_id)

    # Verify the result
    assert result is True

    # multiple DB operations must be performed:
    # 1. check if connection to DB is alive
    # 2. delete one conversation from DB
    calls = [
        call("SELECT 1"),
        call(
            PostgresCache.DELETE_SINGLE_CONVERSATION_STATEMENT,
            (user_id, conversation_id),
        ),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=False)


def test_delete_operation_not_found():
    """Test the Cache.delete operation when the conversation is not found."""
    # Mock the database cursor behavior to simulate no row found
    mock_cursor = MagicMock()
    mock_cursor.rowcount = 0

    # do not use real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # Initialize Postgres cache
        config = PostgresConfig()
        cache = PostgresCache(config)

        # Call the "delete" operation
        result = cache.delete(user_id, conversation_id)

    # Verify the result
    assert result is False

    # multiple DB operations must be performed:
    # 1. check if connection to DB is alive
    # 2. delete one conversation from DB
    calls = [
        call("SELECT 1"),
        call(
            PostgresCache.DELETE_SINGLE_CONVERSATION_STATEMENT,
            (user_id, conversation_id),
        ),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=False)


def test_delete_operation_on_exception():
    """Test the Cache.delete operation when an exception is raised."""
    # Mock the database cursor behavior to raise an exception
    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = psycopg2.DatabaseError("PLSQL error")

    # do not use real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # Initialize Postgres cache
        config = PostgresConfig()
        cache = PostgresCache(config)

        # Verify that the exception is raised
        with pytest.raises(psycopg2.DatabaseError, match="PLSQL error"):
            cache.delete(user_id, conversation_id)


def test_delete_operation_on_disconnected_db():
    """Test the Cache.delete operation when DB is not connected."""
    # mock the query
    mock_cursor = MagicMock()
    mock_cursor.rowcount = 0

    # do not use real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # initialize Postgres cache
        config = PostgresConfig()
        cache = PostgresCache(config)
        # simulate DB disconnection
        cache.connection = None
        assert not cache.connected()
        # DB operation should connect automatically
        cache.delete(user_id, conversation_id)
        assert cache.connected()

    # one DB operations must be performed:
    # 1. delete one conversation from DB
    calls = [
        call(
            PostgresCache.DELETE_SINGLE_CONVERSATION_STATEMENT,
            (user_id, conversation_id),
        ),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=False)


def test_cleanup_method_when_clean_not_needed():
    """Test the static method that cleans up PG cache."""
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (200,)
    capacity = 1000

    # do not use real PostgreSQL instance
    with patch("psycopg2.connect"):
        PostgresCache._cleanup(mock_cursor, capacity)

    # Verify the query execution
    mock_cursor.execute.assert_called_once_with(PostgresCache.QUERY_TOTAL_ENTRIES)


def test_cleanup_method_when_clean_performed():
    """Test the static method that cleans up PG cache by evicting one message."""
    # Prepare mock data for eviction: a conversation with 2 messages
    value = [cache_entry_1.to_dict(), cache_entry_2.to_dict()]
    conversation = json.dumps(value, cls=MessageEncoder)
    value_bytes = conversation.encode("utf-8")
    row = (user_id, conversation_id, value_bytes)

    # After evicting the oldest message, the conversation has 1 message left
    trimmed_value = [cache_entry_2.to_dict()]
    trimmed_conversation = json.dumps(trimmed_value, cls=MessageEncoder)
    trimmed_value_bytes = trimmed_conversation.encode("utf-8")

    mock_cursor = MagicMock()
    mock_cursor.fetchone.side_effect = [(200,), row]
    capacity = 199  # Total 200 > 199, so evict 1 message

    # do not use real PostgreSQL instance
    with patch("psycopg2.connect"):
        PostgresCache._cleanup(mock_cursor, capacity)

    # Verify the query executions: get total, get oldest row, update the trimmed conversation
    calls = [
        call(PostgresCache.QUERY_TOTAL_ENTRIES),
        call(PostgresCache.SELECT_OLDEST_ROW),
        call(
            PostgresCache.UPDATE_CONVERSATION_HISTORY_STATEMENT,
            (trimmed_value_bytes, user_id, conversation_id),
        ),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=False)


def test_ready():
    """Test the Cache.ready operation."""
    # do not use real PostgreSQL instance
    with patch("psycopg2.connect"):
        # initialize Postgres cache
        config = PostgresConfig()
        cache = PostgresCache(config)

        # mock the connection state 0 - open
        cache.connection.closed = 0
        # patch the poll function to return POLL_OK
        cache.connection.poll = MagicMock(return_value=psycopg2.extensions.POLL_OK)
        # cache is ready
        assert cache.ready()

        # mock the connection state 1 - closed
        cache.connection.closed = 1
        # cache is not ready
        assert not cache.ready()

        for error_type in (psycopg2.OperationalError, psycopg2.InterfaceError):
            # mock the connection state 0 - open
            cache.connection.closed = 0
            # patch the poll function to raise OperationalError
            cache.connection.poll = MagicMock(
                side_effect=error_type("Connection closed")
            )
            # cache is not ready
            assert not cache.ready()
