"""Unit tests for PostgresCache transaction management fix."""

from collections.abc import Generator
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


@pytest.fixture(autouse=True)
def _suppress_health_loop() -> Generator[None, None, None]:
    """Prevent the background health-check thread from making real DB calls."""
    with patch.object(PostgresCache, "_health_check_loop"):
        yield


user_id = suid.get_suid()
conversation_id = suid.get_suid()
cache_entry = CacheEntry(
    query=HumanMessage("test message"), response=AIMessage("test response")
)


def test_insert_or_append_transaction_status_check_on_success() -> None:
    """Test that transaction status is checked before setting autocommit on success."""
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None

    with patch("psycopg2.connect") as mock_connect:
        mock_connection = mock_connect.return_value
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connection.get_transaction_status.return_value = (
            psycopg2.extensions.TRANSACTION_STATUS_IDLE
        )

        config = PostgresConfig()
        cache = PostgresCache(config)

        cache.insert_or_append(user_id, conversation_id, cache_entry)

    mock_connection.get_transaction_status.assert_called()
    mock_connection.commit.assert_called()
    assert mock_connection.autocommit is True


def test_insert_or_append_transaction_status_check_on_error() -> None:
    """Test that active transaction is rolled back before setting autocommit on error."""
    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = [
        None,
        psycopg2.DatabaseError("test error"),
    ]

    with patch("psycopg2.connect") as mock_connect:
        mock_connection = mock_connect.return_value
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connection.get_transaction_status.return_value = (
            psycopg2.extensions.TRANSACTION_STATUS_INERROR
        )

        config = PostgresConfig()
        cache = PostgresCache(config)

        with pytest.raises(CacheError):
            cache.insert_or_append(user_id, conversation_id, cache_entry)

    mock_connection.get_transaction_status.assert_called()
    assert mock_connection.rollback.call_count == 3
    assert mock_connection.autocommit is True


def test_delete_transaction_status_check_on_success() -> None:
    """Test that transaction status is checked before setting autocommit on delete success."""
    mock_cursor = MagicMock()
    mock_cursor.rowcount = 1

    with patch("psycopg2.connect") as mock_connect:
        mock_connection = mock_connect.return_value
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
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


def test_delete_transaction_status_check_on_error() -> None:
    """Test that active transaction is rolled back before setting autocommit on delete error."""
    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = [
        None,
        psycopg2.DatabaseError("delete failed"),
    ]

    with patch("psycopg2.connect") as mock_connect:
        mock_connection = mock_connect.return_value
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connection.get_transaction_status.return_value = (
            psycopg2.extensions.TRANSACTION_STATUS_INERROR
        )

        config = PostgresConfig()
        cache = PostgresCache(config)

        with pytest.raises(CacheError):
            cache.delete(user_id, conversation_id)

    mock_connection.get_transaction_status.assert_called()
    assert mock_connection.rollback.call_count == 3
    assert mock_connection.autocommit is True


def test_no_extra_rollback_when_transaction_idle() -> None:
    """Test that no extra rollback is called when transaction is already IDLE."""
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None

    with patch("psycopg2.connect") as mock_connect:
        mock_connection = mock_connect.return_value
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connection.get_transaction_status.return_value = (
            psycopg2.extensions.TRANSACTION_STATUS_IDLE
        )

        config = PostgresConfig()
        cache = PostgresCache(config)

        cache.insert_or_append(user_id, conversation_id, cache_entry)

    assert mock_connection.commit.call_count >= 1
    mock_connection.rollback.assert_not_called()


def test_insert_or_append_ambiguous_commit_raises_cache_error() -> None:
    """Test that connection errors during commit raise CacheError, not raw exception."""
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None
    commit_count = [0]

    def commit_fails_on_second_call() -> None:
        commit_count[0] += 1
        if commit_count[0] > 1:
            raise psycopg2.OperationalError("connection lost during commit")

    with patch("psycopg2.connect") as mock_connect:
        mock_connection = mock_connect.return_value
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connection.get_transaction_status.return_value = (
            psycopg2.extensions.TRANSACTION_STATUS_IDLE
        )
        mock_connection.commit.side_effect = commit_fails_on_second_call

        config = PostgresConfig()
        cache = PostgresCache(config)

        with pytest.raises(CacheError, match="ambiguous commit"):
            cache.insert_or_append(user_id, conversation_id, cache_entry)

    assert commit_count[0] == 2


def test_delete_ambiguous_commit_raises_cache_error() -> None:
    """Test that connection errors during commit in delete raise CacheError."""
    mock_cursor = MagicMock()
    mock_cursor.rowcount = 1
    commit_count = [0]

    def commit_fails_on_second_call() -> None:
        commit_count[0] += 1
        if commit_count[0] > 1:
            raise psycopg2.InterfaceError("connection lost during commit")

    with patch("psycopg2.connect") as mock_connect:
        mock_connection = mock_connect.return_value
        mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connection.get_transaction_status.return_value = (
            psycopg2.extensions.TRANSACTION_STATUS_IDLE
        )
        mock_connection.commit.side_effect = commit_fails_on_second_call

        config = PostgresConfig()
        cache = PostgresCache(config)

        with pytest.raises(CacheError, match="ambiguous commit"):
            cache.delete(user_id, conversation_id)

    assert commit_count[0] == 2
