"""Unit tests for runners."""

from unittest.mock import MagicMock, patch

import pytest

from ols import constants
from ols.app.models.config import (
    Config,
    LimiterConfig,
    LimitersConfig,
    PostgresConfig,
    QuotaHandlersConfig,
    SchedulerConfig,
)
from ols.runners.quota_scheduler import (
    INCREASE_QUOTA_STATEMENT,
    RESET_QUOTA_STATEMENT,
    connect,
    get_subject_id,
    increase_quota,
    quota_revocation,
    quota_scheduler,
    reset_quota,
    start_quota_scheduler,
)


def test_quota_scheduler_no_config():
    """Test the quota_scheduler function with empty configuration."""
    # no configuration
    config = None

    # quota scheduler should not start
    assert quota_scheduler(config) is False


def test_quota_scheduler_connection_not_setup():
    """Test the quota_scheduler function when connection is not setup."""
    # default quota handlers configuration
    config = QuotaHandlersConfig()

    # storage configuration is not provided
    config.storage = None

    # don't connect to real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value = None

        # quota scheduler should not start
        assert quota_scheduler(config) is False


def test_quota_scheduler_no_connection():
    """Test the quota_scheduler function when connection can not be established."""
    # default quota handlers configuration
    config = QuotaHandlersConfig()

    # connection won't be established
    config.storage = PostgresConfig()

    # don't connect to real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value = None

        # quota scheduler should not start
        assert quota_scheduler(config) is False


def test_quota_scheduler_no_limiters():
    """Test the quota_scheduler function when limiters are not setup."""
    # default quota handlers configuration
    config = QuotaHandlersConfig()

    # connection won't be established
    config.storage = PostgresConfig()

    # scheduler configuration
    config.scheduler = SchedulerConfig(period=10)

    # don't connect to real PostgreSQL instance
    with patch("psycopg2.connect"):
        # quota scheduler should not start
        assert quota_scheduler(config) is False


def test_quota_scheduler_empty_list_of_limiters():
    """Test the quota_scheduler function when empty list of limiters are setup."""
    # default quota handlers configuration
    config = QuotaHandlersConfig()

    # connection won't be established
    config.storage = PostgresConfig()

    # scheduler configuration
    config.scheduler = SchedulerConfig(period=10)

    # quota limiters configuration
    config.limiters = LimitersConfig()

    # we need to be able to end the endless loop by raising exception
    with (
        patch("psycopg2.connect"),
        patch("ols.runners.quota_scheduler.sleep", side_effect=Exception()),
    ):
        # just try to enter the endless loop
        with pytest.raises(Exception):
            quota_scheduler(config)


def test_quota_scheduler_non_empty_list_of_limiters():
    """Test the quota_scheduler function when non empty list of limiters are setup."""
    # default quota handlers configuration
    config = QuotaHandlersConfig()

    # connection won't be established
    config.storage = PostgresConfig()

    # scheduler configuration
    config.scheduler = SchedulerConfig(period=10)

    # quota limiters configuration
    config.limiters = LimitersConfig(
        [
            {
                "name": "foo",
                "type": "cluster_limiter",
                "initial_quota": 1000,
                "quota_increase": 10,
                "period": "5 days",
            }
        ]
    )

    # we need to be able to end the endless loop by raising exception
    with (
        patch("psycopg2.connect"),
        patch("ols.runners.quota_scheduler.sleep", side_effect=Exception()),
    ):
        # just try to enter the endless loop
        with pytest.raises(Exception):
            quota_scheduler(config)


def test_quota_scheduler_limiter_without_type():
    """Test the quota_scheduler function when limiter type is not specified."""
    # default quota handlers configuration
    config = QuotaHandlersConfig()

    # connection won't be established
    config.storage = PostgresConfig()

    # scheduler configuration
    config.scheduler = SchedulerConfig(period=10)

    # quota limiters configuration
    config.limiters = LimitersConfig(
        [
            {
                "name": "foo",
                "type": "cluster_limiter",
                "initial_quota": 1000,
                "quota_increase": 10,
                "period": "5 days",
            }
        ]
    )
    # this will cause quota_revocation function to raise an Exception
    config.limiters.limiters["foo"].type = None

    # we need to be able to end the endless loop by raising exception
    with (
        patch("psycopg2.connect"),
        patch("ols.runners.quota_scheduler.sleep", side_effect=Exception()),
    ):
        # just try to enter the endless loop
        with pytest.raises(Exception):
            quota_scheduler(config)


def test_connect():
    """Test the connection to Postgres."""
    exception_message = "Exception during PostgreSQL storage."

    # connection won't be established
    config = PostgresConfig()

    # don't connect to real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.execute.side_effect = Exception(
            exception_message
        )
        # try to connect to mocked Postgres
        connection = connect(config)

        # connection should not be established
        assert connection is not None


def test_get_subject_id():
    """Check the function to get subject ID based on quota limiter type."""
    assert get_subject_id(constants.USER_QUOTA_LIMITER) == "u"
    assert get_subject_id(constants.CLUSTER_QUOTA_LIMITER) == "c"
    assert get_subject_id("foobar") == "?"


def test_increase_quota():
    """Check the function that increases quota for given subject."""
    # mock the query result - no data
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None

    # connection won't be established
    config = PostgresConfig()

    # parameters for increasing quota
    subject_id = "u"
    increase_by = 10
    period = "5 days"

    # don't connect to real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        # try to connect to mocked Postgres
        connection = connect(config)

        increase_quota(connection, subject_id, increase_by, period)

        # quota should be increased in mocked database
        mock_cursor.execute.assert_called_once_with(
            INCREASE_QUOTA_STATEMENT, (increase_by, subject_id, period)
        )


def test_reset_quota():
    """Check the function that resets quota for given subject."""
    # mock the query result - no data
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None

    # connection won't be established
    config = PostgresConfig()

    # parameters for resetting quota
    subject_id = "u"
    reset_to = 1000
    period = "5 days"

    # don't connect to real PostgreSQL instance
    with patch("psycopg2.connect") as mock_connect:
        # try to connect to mocked Postgres
        connection = connect(config)
        mock_connect.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )
        reset_quota(connection, subject_id, reset_to, period)

        # quota should be reset in mocked database
        mock_cursor.execute.assert_called_once_with(
            RESET_QUOTA_STATEMENT, (reset_to, subject_id, period)
        )


def test_quota_revocation_no_limiter_type():
    """Test the quota_revocation function when limiter type is not specified."""
    # quota limiter configuration
    quota_limiter = LimiterConfig(
        type=None, initial_quota=10, quota_increase=10, period="3 days"
    )

    # exception should be raised
    expected = "Limiter type not set, skipping revocation"
    with pytest.raises(Exception, match=expected):
        quota_revocation(None, "u", quota_limiter)


def test_quota_revocation_no_limiter_period():
    """Test the quota_revocation function when limiter period is not specified."""
    # quota limiter configuration
    quota_limiter = LimiterConfig(
        type=constants.USER_QUOTA_LIMITER,
        initial_quota=10,
        quota_increase=10,
        period=None,
    )

    # exception should be raised
    expected = "Limiter period not set, skipping revocation"
    with pytest.raises(Exception, match=expected):
        quota_revocation(None, "u", quota_limiter)


@pytest.fixture
def default_config():
    """Fixture providing default configuration."""
    return Config(
        {
            "llm_providers": [],
            "ols_config": {
                "default_provider": "test_default_provider",
                "default_model": "test_default_model",
                "conversation_cache": {
                    "type": "memory",
                    "memory": {
                        "max_entries": 100,
                    },
                },
                "logging_config": {
                    "app_log_level": "error",
                },
                "query_validation_method": "disabled",
                "certificate_directory": "/foo/bar/baz",
                "authentication_config": {"module": "foo"},
                "limiters": {},
            },
            "dev_config": {"disable_tls": "true"},
        }
    )


def test_start_quota_scheduler(default_config):
    """Test the function that starts quota scheduler."""
    # do not really start a new thread, just mock the creation
    with patch("ols.runners.quota_scheduler.Thread"):
        # should not raise any exception
        start_quota_scheduler(default_config)
