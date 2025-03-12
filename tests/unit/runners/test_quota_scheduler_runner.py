"""Unit tests for runners."""

from unittest.mock import MagicMock, patch

import pytest

from ols import constants
from ols.app.models.config import (
    Config,
    LimiterConfig,
    LimitersConfig,
    PostgresConfig,
    QuotaLimiterConfig,
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


@patch("psycopg2.connect")
def test_quota_scheduler_connection_not_setup(mock_connect):
    """Test the quota_scheduler function when connection is not setup."""
    mock_connect.return_value = None
    config = QuotaLimiterConfig()

    # storage configuration is not provided
    config.storage = None

    # quota scheduler should not start
    assert quota_scheduler(config) is False


@patch("psycopg2.connect")
def test_quota_scheduler_no_connection(mock_connect):
    """Test the quota_scheduler function when connection can not be established."""
    mock_connect.return_value = None
    config = QuotaLimiterConfig()

    # connection won't be established
    config.storage = PostgresConfig()

    # quota scheduler should not start
    assert quota_scheduler(config) is False


@patch("psycopg2.connect")
def test_quota_scheduler_no_limiters(mock_connect):
    """Test the quota_scheduler function when limiters are not setup."""
    config = QuotaLimiterConfig()
    config.storage = PostgresConfig()
    config.scheduler = SchedulerConfig(period=10)

    # quota scheduler should not start
    assert quota_scheduler(config) is False


@patch("psycopg2.connect")
def test_quota_scheduler_empty_list_of_limiters(mock_connect):
    """Test the quota_scheduler function when empty list of limiters are setup."""
    config = QuotaLimiterConfig()
    config.storage = PostgresConfig()
    config.scheduler = SchedulerConfig(period=10)
    config.limiters = LimitersConfig()

    # we need to be able to end the endless loop by raising exception
    with patch("ols.runners.quota_scheduler.sleep", side_effect=Exception()):
        # just try to enter the endless loop
        with pytest.raises(Exception):
            quota_scheduler(config)


@patch("psycopg2.connect")
def test_quota_scheduler_non_empty_list_of_limiters(mock_connect):
    """Test the quota_scheduler function when non empty list of limiters are setup."""
    config = QuotaLimiterConfig()
    config.storage = PostgresConfig()
    config.scheduler = SchedulerConfig(period=10)
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
    with patch("ols.runners.quota_scheduler.sleep", side_effect=Exception()):
        # just try to enter the endless loop
        with pytest.raises(Exception):
            quota_scheduler(config)


@patch("psycopg2.connect")
def test_quota_scheduler_limiter_without_type(mock_connect):
    """Test the quota_scheduler function when limiter type is not specified."""
    config = QuotaLimiterConfig()
    config.storage = PostgresConfig()
    config.scheduler = SchedulerConfig(period=10)
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
    with patch("ols.runners.quota_scheduler.sleep", side_effect=Exception()):
        # just try to enter the endless loop
        with pytest.raises(Exception):
            quota_scheduler(config)


@patch("psycopg2.connect")
def test_connect(mock_connect):
    """Test the connection to Postgres."""
    exception_message = "Exception during PostgreSQL storage."
    mock_connect.return_value.cursor.return_value.execute.side_effect = Exception(
        exception_message
    )

    # try to connect to mocked Postgres
    config = PostgresConfig()
    connection = connect(config)

    # connection should not be established
    assert connection is not None


def test_get_subject_id():
    """Check the function to get subject ID based on quota limiter type."""
    assert get_subject_id(constants.USER_QUOTA_LIMITER) == "u"
    assert get_subject_id(constants.CLUSTER_QUOTA_LIMITER) == "c"
    assert get_subject_id("foobar") == "?"


@patch("psycopg2.connect")
def test_increase_quota(mock_connect):
    """Check the function that increases quota for given subject."""
    # mock the query result - no data
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None
    mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

    # try to connect to mocked Postgres
    config = PostgresConfig()
    connection = connect(config)

    subject_id = "u"
    increase_by = 10
    period = "5 days"
    increase_quota(connection, subject_id, increase_by, period)

    # quota should be increased in mocked database
    mock_cursor.execute.assert_called_once_with(
        INCREASE_QUOTA_STATEMENT, (increase_by, subject_id, period)
    )


@patch("psycopg2.connect")
def test_reset_quota(mock_connect):
    """Check the function that resets quota for given subject."""
    # mock the query result - no data
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None
    mock_connect.return_value.cursor.return_value.__enter__.return_value = mock_cursor

    # try to connect to mocked Postgres
    config = PostgresConfig()
    connection = connect(config)

    subject_id = "u"
    reset_to = 1000
    period = "5 days"
    reset_quota(connection, subject_id, reset_to, period)

    # quota should be reset in mocked database
    mock_cursor.execute.assert_called_once_with(
        RESET_QUOTA_STATEMENT, (reset_to, subject_id, period)
    )


def test_quota_revocation_no_limiter_type():
    """Test the quota_revocation function when limiter type is not specified."""
    quota_limiter = LimiterConfig(
        type=None, initial_quota=10, quota_increase=10, period="3 days"
    )

    # exception should be raised
    expected = "Limiter type not set, skipping revocation"
    with pytest.raises(Exception, match=expected):
        quota_revocation(None, "u", quota_limiter)


def test_quota_revocation_no_limiter_period():
    """Test the quota_revocation function when limiter period is not specified."""
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
