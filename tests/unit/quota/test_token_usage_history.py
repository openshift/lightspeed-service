"""Unit tests for TokenUsageHistory class."""

import datetime
from unittest.mock import MagicMock, patch

import pytest

from ols.app.models.config import PostgresConfig
from ols.src.quota.token_usage_history import TokenUsageHistory


def test_init_storage_failure_detection():
    """Test the exception handling for storage initialize operation."""
    exception_message = "Exception during PostgreSQL storage."

    # do not use connection to real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.execute.side_effect = Exception(
            exception_message
        )

        # try to connect to mocked Postgres
        config = PostgresConfig()
        with pytest.raises(Exception, match=exception_message):
            TokenUsageHistory(config)

        # connection must be closed in case of exception
        mock_connect.return_value.close.assert_called_once_with()


@patch("ols.src.quota.token_usage_history.datetime")
def test_consume_tokens(mock_datetime):
    """Test the operation to consume tokens."""
    input_tokens = 10
    output_tokens = 20
    user_id = "1234"
    provider = "X"
    model = "Y"

    # mock the query result - no data
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (None,)

    # do not use connection to real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # mock for real timestamp
        timestamp = datetime.datetime(2000, 1, 1, 12, 0, 0)

        # mock function to retrieve timestamp
        mock_datetime.now = lambda: timestamp

        # initialize Postgres storage
        config = PostgresConfig()
        q = TokenUsageHistory(config)

        # try to consume tokens
        q.consume_tokens(user_id, provider, model, input_tokens, output_tokens)

    mock_cursor.execute.assert_called_once_with(
        TokenUsageHistory.CONSUME_TOKENS_FOR_USER,
        {
            "user_id": user_id,
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "updated_at": timestamp,
        },
    )
