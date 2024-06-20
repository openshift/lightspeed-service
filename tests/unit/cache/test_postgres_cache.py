"""Unit tests for PostgresCache class."""

import json
from unittest.mock import MagicMock, call, patch

import psycopg2
import pytest

from ols.app.endpoints.ols import ai_msg, human_msg
from ols.app.models.config import PostgresConfig
from ols.src.cache.cache_error import CacheError
from ols.src.cache.postgres_cache import PostgresCache
from ols.utils import suid

user_id = suid.get_suid()
conversation_id = suid.get_suid()


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
    assert conversation is None
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
        human_msg("first message from human"),
        ai_msg("first answer from AI"),
        human_msg("second message from human"),
        ai_msg("second answer from AI"),
    ]

    conversation = json.dumps(history)

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
        value = cache.get(user_id, conversation_id)
        assert value is None


@patch("psycopg2.connect")
def test_insert_or_append_operation_first_item(mock_connect):
    """Test the Cache.insert_or_append operation for first item to be inserted."""
    history = [
        human_msg("first message from human"),
        ai_msg("first answer from AI"),
        human_msg("second message from human"),
        ai_msg("second answer from AI"),
    ]

    conversation = json.dumps(history)

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
            (user_id, conversation_id, conversation),
        ),
        call(PostgresCache.QUERY_CACHE_SIZE),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=True)


@patch("psycopg2.connect")
def test_insert_or_append_operation_append_item(mock_connect):
    """Test the Cache.insert_or_append operation for more item to be inserted."""
    stored_history = [
        human_msg("first message from human"),
        ai_msg("first answer from AI"),
    ]

    old_conversation = json.dumps(stored_history)

    appended_history = [
        human_msg("first message from human"),
        ai_msg("first answer from AI"),
    ]

    # create jsond object in the exactly same format
    whole_history = json.loads(old_conversation)
    whole_history.extend(appended_history)
    new_conversation = json.dumps(whole_history)

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
            (new_conversation, user_id, conversation_id),
        ),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=True)


@patch("psycopg2.connect")
def test_insert_or_append_operation_on_exception(mock_connect):
    """Test the Cache.insert_or_append operation when exception is thrown."""
    history = [
        human_msg("first message from human"),
        ai_msg("first answer from AI"),
        human_msg("second message from human"),
        ai_msg("second answer from AI"),
    ]

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
