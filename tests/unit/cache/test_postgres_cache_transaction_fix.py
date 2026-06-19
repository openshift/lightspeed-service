"""Unit tests for PostgresCache transaction management fix."""

from unittest.mock import MagicMock, patch

import psycopg2
import psycopg2.extensions
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ols.app.models.config import PostgresConfig
from ols.app.models.models import CacheEntry
from ols.src.cache.cache_error import CacheError
from ols.src.cache.postgres_cache import PostgresCache
from ols.utils import suid

user_id = suid.get_suid()
conversation_id = suid.get_suid()
cache_entry = CacheEntry(
    query=HumanMessage("test message"), response=AIMessage("test response")
)


def test_insert_or_append_transaction_status_check_on_success():
    """Test that transaction status is checked before setting autocommit on success."""
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None

    with patch("psycopg2.connect") as mock_connect:
        # Mock connection with proper transaction status
        mock_connection = mock_connect.return_value
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        # After successful commit, transaction should be IDLE
        mock_connection.get_transaction_status.return_value = (
            psycopg2.extensions.TRANSACTION_STATUS_IDLE
        )

        config = PostgresConfig()
        cache = PostgresCache(config)

        cache.insert_or_append(user_id, conversation_id, cache_entry)

    # Verify transaction status was checked
    mock_connection.get_transaction_status.assert_called()
    # Commit should be called (successful operation)
    mock_connection.commit.assert_called()
    # Rollback should NOT be called in finally (transaction already IDLE)
    # Note: rollback might be called in except block, but not in finally
    assert mock_connection.autocommit is True


def test_insert_or_append_transaction_status_check_on_error():
    """Test that active transaction is rolled back before setting autocommit on error."""
    mock_cursor = MagicMock()
    # Simulate database error
    mock_cursor.execute.side_effect = [
        None,  # SELECT 1 (connection check)
        psycopg2.DatabaseError("test error"),  # advisory lock fails
    ]

    with patch("psycopg2.connect") as mock_connect:
        mock_connection = mock_connect.return_value
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        # After error, transaction is still ACTIVE (not IDLE)
        mock_connection.get_transaction_status.return_value = (
            psycopg2.extensions.TRANSACTION_STATUS_INERROR
        )

        config = PostgresConfig()
        cache = PostgresCache(config)

        with pytest.raises(CacheError):
            cache.insert_or_append(user_id, conversation_id, cache_entry)

    # Verify transaction status was checked in finally
    mock_connection.get_transaction_status.assert_called()
    # Rollback should be called three times:
    # 1. Before disabling autocommit (transaction is not IDLE)
    # 2. In the except block
    # 3. In the finally block because transaction is not IDLE
    assert mock_connection.rollback.call_count == 3
    assert mock_connection.autocommit is True


def test_delete_transaction_status_check_on_success():
    """Test that transaction status is checked before setting autocommit on delete success."""
    mock_cursor = MagicMock()
    mock_cursor.rowcount = 1  # Simulate successful delete

    with patch("psycopg2.connect") as mock_connect:
        mock_connection = mock_connect.return_value
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        # After successful commit, transaction should be IDLE
        mock_connection.get_transaction_status.return_value = (
            psycopg2.extensions.TRANSACTION_STATUS_IDLE
        )

        config = PostgresConfig()
        cache = PostgresCache(config)

        result = cache.delete(user_id, conversation_id)

    assert result is True
    mock_connection.get_transaction_status.assert_called()
    mock_connection.commit.assert_called()
    assert mock_connection.autocommit is True


def test_delete_transaction_status_check_on_error():
    """Test that active transaction is rolled back before setting autocommit on delete error."""
    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = [
        None,  # SELECT 1 (connection check)
        psycopg2.DatabaseError("delete failed"),  # delete fails
    ]

    with patch("psycopg2.connect") as mock_connect:
        mock_connection = mock_connect.return_value
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        # After error, transaction is still ACTIVE
        mock_connection.get_transaction_status.return_value = (
            psycopg2.extensions.TRANSACTION_STATUS_INERROR
        )

        config = PostgresConfig()
        cache = PostgresCache(config)

        with pytest.raises(CacheError):
            cache.delete(user_id, conversation_id)

    mock_connection.get_transaction_status.assert_called()
    # Rollback called three times: before disabling autocommit, in except, and in finally
    assert mock_connection.rollback.call_count == 3
    assert mock_connection.autocommit is True


def test_no_extra_rollback_when_transaction_idle():
    """Test that no extra rollback is called when transaction is already IDLE."""
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None

    with patch("psycopg2.connect") as mock_connect:
        mock_connection = mock_connect.return_value
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        # Transaction is IDLE (normal successful case)
        mock_connection.get_transaction_status.return_value = (
            psycopg2.extensions.TRANSACTION_STATUS_IDLE
        )

        config = PostgresConfig()
        cache = PostgresCache(config)

        cache.insert_or_append(user_id, conversation_id, cache_entry)

    # In successful case with IDLE transaction:
    # - commit() is called (during init and during insert_or_append)
    # - rollback() should NOT be called in finally (transaction is IDLE)
    assert mock_connection.commit.call_count >= 1
    # Rollback should not be called at all in the success case
    mock_connection.rollback.assert_not_called()
