"""Unit tests for ClusterQuotaLimiter class."""

import datetime
from unittest.mock import MagicMock, call, patch

import pytest

from ols.app.models.config import PostgresConfig
from ols.src.quota.cluster_quota_limiter import ClusterQuotaLimiter


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
            ClusterQuotaLimiter(config, 0)

        # connection must be closed in case of exception
        mock_connect.return_value.close.assert_called_once_with()


def test_init_quota():
    """Test the init quota operation."""
    quota_limit = 100
    subject = "c"

    # mock the query result - with empty storage
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None

    # mock for real timestamp
    timestamp = datetime.datetime(2000, 1, 1, 12, 0, 0)

    # do not use connection to real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # mock the datetime class in order to use constant timestamps
        with patch("ols.src.quota.revokable_quota_limiter.datetime") as mock_datetime:
            # mock function to retrieve timestamp
            mock_datetime.now = lambda: timestamp

            # initialize Postgres storage
            config = PostgresConfig()
            q = ClusterQuotaLimiter(config, quota_limit)

            # init quota for given cluster
            q._init_quota()

    # new record should be inserted into storage
    mock_cursor.execute.assert_called_once_with(
        ClusterQuotaLimiter.INIT_QUOTA,
        ("", subject, quota_limit, quota_limit, timestamp),
    )


def test_available_quota_with_data():
    """Test the get available quota operation."""
    quota_limit = 100
    available_quota = 50
    subject = "c"

    # mock the query result - available data in the table
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (available_quota,)

    # do not use connection to real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # initialize Postgres storage
        config = PostgresConfig()
        q = ClusterQuotaLimiter(config, quota_limit)

        # try to retrieve available quota for given cluster
        available = q.available_quota()

    # expected calls to storage
    calls = [
        # check if storage connection is alive
        call("SELECT 1"),
        # quota for given cluster should be read from storage
        call(ClusterQuotaLimiter.SELECT_QUOTA, ("", subject)),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=False)
    assert available == available_quota


def test_available_quota_no_data():
    """Test the get available quota operation."""
    quota_limit = 100
    subject = "c"

    # mock the query result - no data
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None

    # mock for real timestamp
    timestamp = datetime.datetime(2000, 1, 1, 12, 0, 0)

    # do not use connection to real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # mock the datetime class in order to use constant timestamps
        with patch("ols.src.quota.revokable_quota_limiter.datetime") as mock_datetime:
            # mock function to retrieve timestamp
            mock_datetime.now = lambda: timestamp

            # initialize Postgres storage
            config = PostgresConfig()
            q = ClusterQuotaLimiter(config, quota_limit)

            # try to retrieve available quota for given cluster
            available = q.available_quota()

    # quota for given cluster should be read from storage
    # and the initialization of new record should be made
    calls = [
        call(ClusterQuotaLimiter.SELECT_QUOTA, ("", subject)),
        call(
            ClusterQuotaLimiter.INIT_QUOTA,
            ("", subject, quota_limit, quota_limit, timestamp),
        ),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=False)
    assert available == quota_limit


def test_available_quota_on_disconnected_db():
    """Test the get available quota operation when DB is not connected."""
    quota_limit = 100

    # mock the query result - no data
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None

    # mock for real timestamp
    timestamp = datetime.datetime(2000, 1, 1, 12, 0, 0)

    # do not use connection to real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # mock the datetime class in order to use constant timestamps
        with patch("ols.src.quota.revokable_quota_limiter.datetime") as mock_datetime:
            # mock function to retrieve timestamp
            mock_datetime.now = lambda: timestamp

            # initialize Postgres storage
            config = PostgresConfig()
            q = ClusterQuotaLimiter(config, quota_limit)

            # simulate DB disconnection
            q.connection = None
            assert not q.connected()

            # try to retrieve available quota for given cluster
            available = q.available_quota()

            # DB operation should connect automatically
            assert q.connected()

    assert available == quota_limit


def test_revoke_quota():
    """Test the operation to revoke quota."""
    quota_limit = 100
    subject = "c"

    # mock the query result - no data
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None

    # mock for real timestamp
    timestamp = datetime.datetime(2000, 1, 1, 12, 0, 0)

    # do not use connection to real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # mock the datetime class in order to use constant timestamps
        with patch("ols.src.quota.revokable_quota_limiter.datetime") as mock_datetime:
            # mock function to retrieve timestamp
            mock_datetime.now = lambda: timestamp

            # initialize Postgres storage
            config = PostgresConfig()
            q = ClusterQuotaLimiter(config, quota_limit)

            # try to revoke quota
            q.revoke_quota()

    # expected calls to storage
    calls = [
        # check if storage connection is alive
        call("SELECT 1"),
        # quota for given cluster should be written into the storage
        call(
            ClusterQuotaLimiter.SET_AVAILABLE_QUOTA,
            (quota_limit, timestamp, "", subject),
        ),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=False)


def test_revoke_quota_on_disconnected_db():
    """Test the operation to revoke quota when DB is not connected."""
    quota_limit = 100

    # mock the query result - no data
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None

    # mock for real timestamp
    timestamp = datetime.datetime(2000, 1, 1, 12, 0, 0)

    # do not use connection to real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # mock the datetime class in order to use constant timestamps
        with patch("ols.src.quota.revokable_quota_limiter.datetime") as mock_datetime:
            # mock function to retrieve timestamp
            mock_datetime.now = lambda: timestamp

            # initialize Postgres storage
            config = PostgresConfig()
            q = ClusterQuotaLimiter(config, quota_limit)

            # simulate DB disconnection
            q.connection = None
            assert not q.connected()

            # try to revoke quota
            q.revoke_quota()

            # DB operation should connect automatically
            assert q.connected()


def test_consume_tokens_not_enough():
    """Test the operation to consume tokens."""
    to_be_consumed = 100
    available_tokens = 50
    quota_limit = 100
    subject = "c"

    # mock the query result - no data
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (available_tokens,)

    # mock for real timestamp
    timestamp = datetime.datetime(2000, 1, 1, 12, 0, 0)

    # do not use connection to real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # mock the datetime class in order to use constant timestamps
        with patch("ols.src.quota.revokable_quota_limiter.datetime") as mock_datetime:
            # mock function to retrieve timestamp
            mock_datetime.now = lambda: timestamp

            # initialize Postgres storage
            config = PostgresConfig()
            q = ClusterQuotaLimiter(config, quota_limit)

            # try to consume tokens
            q.consume_tokens(to_be_consumed, 0)

    # expected calls to storage
    calls = [
        # check if storage connection is alive
        call("SELECT 1"),
        # quota for given user should be updated in storage
        call(
            ClusterQuotaLimiter.UPDATE_AVAILABLE_QUOTA,
            (-to_be_consumed, timestamp, "", subject),
        ),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=False)


def test_consume_input_tokens_enough_tokens():
    """Test the operation to consume tokens."""
    to_be_consumed = 50
    available_tokens = 100
    quota_limit = 100
    subject = "c"

    # mock the query result - no data
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (available_tokens,)

    # mock for real timestamp
    timestamp = datetime.datetime(2000, 1, 1, 12, 0, 0)

    # do not use connection to real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # mock the datetime class in order to use constant timestamps
        with patch("ols.src.quota.revokable_quota_limiter.datetime") as mock_datetime:
            # mock function to retrieve timestamp
            mock_datetime.now = lambda: timestamp

            # initialize Postgres storage
            config = PostgresConfig()
            q = ClusterQuotaLimiter(config, quota_limit)

            # try to consume tokens
            q.consume_tokens(to_be_consumed, 0)

    # expected calls to storage
    calls = [
        # check if storage connection is alive
        call("SELECT 1"),
        # quota for given user should be updated in storage
        call(
            ClusterQuotaLimiter.UPDATE_AVAILABLE_QUOTA,
            (-to_be_consumed, timestamp, "", subject),
        ),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=False)


def test_consume_output_tokens_enough_tokens():
    """Test the operation to consume tokens."""
    to_be_consumed = 50
    available_tokens = 100
    quota_limit = 100
    subject = "c"

    # mock the query result - no data
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (available_tokens,)

    # mock for real timestamp
    timestamp = datetime.datetime(2000, 1, 1, 12, 0, 0)

    # do not use connection to real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # mock the datetime class in order to use constant timestamps
        with patch("ols.src.quota.revokable_quota_limiter.datetime") as mock_datetime:
            # mock function to retrieve timestamp
            mock_datetime.now = lambda: timestamp

            # initialize Postgres storage
            config = PostgresConfig()
            q = ClusterQuotaLimiter(config, quota_limit)

            # try to consume tokens
            q.consume_tokens(0, to_be_consumed)

    # expected calls to storage
    calls = [
        # check if storage connection is alive
        call("SELECT 1"),
        # quota for given user should be updated in storage
        call(
            ClusterQuotaLimiter.UPDATE_AVAILABLE_QUOTA,
            (-to_be_consumed, timestamp, "", subject),
        ),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=False)


def test_consume_input_and_output_tokens_enough_tokens():
    """Test the operation to consume tokens."""
    input_tokens = 30
    output_tokens = 20
    to_be_consumed = input_tokens + output_tokens
    available_tokens = 100
    quota_limit = 100
    subject = "c"

    # mock the query result - no data
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (available_tokens,)

    # mock for real timestamp
    timestamp = datetime.datetime(2000, 1, 1, 12, 0, 0)

    # do not use connection to real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # mock the datetime class in order to use constant timestamps
        with patch("ols.src.quota.revokable_quota_limiter.datetime") as mock_datetime:
            # mock function to retrieve timestamp
            mock_datetime.now = lambda: timestamp

            # initialize Postgres storage
            config = PostgresConfig()
            q = ClusterQuotaLimiter(config, quota_limit)

            # try to consume tokens
            q.consume_tokens(input_tokens, output_tokens)

    # expected calls to storage
    calls = [
        # check if storage connection is alive
        call("SELECT 1"),
        # quota for given user should be updated in storage
        call(
            ClusterQuotaLimiter.UPDATE_AVAILABLE_QUOTA,
            (-to_be_consumed, timestamp, "", subject),
        ),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=False)


def test_consume_tokens_on_no_record():
    """Test the operation to consume tokens."""
    to_be_consumed = 100
    quota_limit = 100
    subject = "c"

    # mock the query result - no data
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None

    # mock for real timestamp
    timestamp = datetime.datetime(2000, 1, 1, 12, 0, 0)

    # do not use connection to real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # mock the datetime class in order to use constant timestamps
        with patch("ols.src.quota.revokable_quota_limiter.datetime") as mock_datetime:
            # mock function to retrieve timestamp
            mock_datetime.now = lambda: timestamp

            # initialize Postgres storage
            config = PostgresConfig()
            q = ClusterQuotaLimiter(config, quota_limit)

            q.consume_tokens(to_be_consumed, 0)

    # expected calls to storage
    calls = [
        # check if storage connection is alive
        call("SELECT 1"),
        # quota for given user should be updated in storage
        call(
            ClusterQuotaLimiter.UPDATE_AVAILABLE_QUOTA,
            (-to_be_consumed, timestamp, "", subject),
        ),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=False)


def test_consume_tokens_on_disconnected_db():
    """Test the operation to consume tokens when DB is disconnected."""
    input_tokens = 30
    output_tokens = 20
    available_tokens = 100
    quota_limit = 100

    # mock the query result - no data
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (available_tokens,)

    # mock for real timestamp
    timestamp = datetime.datetime(2000, 1, 1, 12, 0, 0)

    # do not use connection to real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # mock the datetime class in order to use constant timestamps
        with patch("ols.src.quota.revokable_quota_limiter.datetime") as mock_datetime:
            # mock function to retrieve timestamp
            mock_datetime.now = lambda: timestamp

            # initialize Postgres storage
            config = PostgresConfig()
            q = ClusterQuotaLimiter(config, quota_limit)

            # simulate DB disconnection
            q.connection = None
            assert not q.connected()

            # try to consume tokens
            q.consume_tokens(input_tokens, output_tokens)

            # DB operation should connect automatically
            assert q.connected()


def test_increase_quota():
    """Test the operation to increase quota."""
    quota_limit = 100
    additional_quota = 10
    subject = "c"

    # mock the query result - no data
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None

    # mock for real timestamp
    timestamp = datetime.datetime(2000, 1, 1, 12, 0, 0)

    # do not use connection to real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # mock the datetime class in order to use constant timestamps
        with patch("ols.src.quota.revokable_quota_limiter.datetime") as mock_datetime:
            # mock function to retrieve timestamp
            mock_datetime.now = lambda: timestamp

            # initialize Postgres storage
            config = PostgresConfig()
            q = ClusterQuotaLimiter(config, quota_limit, additional_quota)

            # try to increase quota
            q.increase_quota()

    # expected calls to storage
    calls = [
        # check if storage connection is alive
        call("SELECT 1"),
        # quota for given cluster should be written into the storage
        call(
            ClusterQuotaLimiter.UPDATE_AVAILABLE_QUOTA,
            (additional_quota, timestamp, "", subject),
        ),
    ]
    mock_cursor.execute.assert_has_calls(calls, any_order=False)


def test_increase_quota_on_disconnected_db():
    """Test the operation to increase quota when DB is not connected."""
    quota_limit = 100

    # mock the query result - no data
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None

    # mock for real timestamp
    timestamp = datetime.datetime(2000, 1, 1, 12, 0, 0)

    # do not use connection to real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # mock the datetime class in order to use constant timestamps
        with patch("ols.src.quota.revokable_quota_limiter.datetime") as mock_datetime:
            # mock function to retrieve timestamp
            mock_datetime.now = lambda: timestamp

            # initialize Postgres storage
            config = PostgresConfig()
            q = ClusterQuotaLimiter(config, quota_limit)

            # simulate DB disconnection
            q.connection = None
            assert not q.connected()

            # try to increase quota
            q.increase_quota()

            # DB operation should connect automatically
            assert q.connected()


def test_ensure_available_quota():
    """Test the operation to increase quota."""
    quota_limit = 0
    additional_quota = 0

    # mock the query result - no data
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None

    # mock for real timestamp
    timestamp = datetime.datetime(2000, 1, 1, 12, 0, 0)

    # do not use connection to real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # mock the datetime class in order to use constant timestamps
        with patch("ols.src.quota.revokable_quota_limiter.datetime") as mock_datetime:
            # mock function to retrieve timestamp
            mock_datetime.now = lambda: timestamp

            # initialize Postgres storage
            config = PostgresConfig()
            q = ClusterQuotaLimiter(config, quota_limit, additional_quota)

            exception_message = "Cluster has no available tokens"

            # check available quota
            with pytest.raises(Exception, match=exception_message):
                q.ensure_available_quota()
