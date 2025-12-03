"""Unit tests for quota limiter factory class."""

from unittest.mock import patch

import pytest

from ols.app.models.config import LimitersConfig, PostgresConfig, QuotaHandlersConfig
from ols.src.quota.cluster_quota_limiter import ClusterQuotaLimiter
from ols.src.quota.quota_limiter_factory import QuotaLimiterFactory
from ols.src.quota.user_quota_limiter import UserQuotaLimiter


def test_quota_limiters_no_storage():
    """Test the quota limiters creating when no storage is configured."""
    config = QuotaHandlersConfig()
    config.storage = None
    limiters = QuotaLimiterFactory.quota_limiters(config)
    assert limiters == []


def test_quota_limiters_no_limiters():
    """Test the quota limiters creating when no limiters are specified."""
    config = QuotaHandlersConfig()
    config.storage = PostgresConfig()
    config.limiters = None
    limiters = QuotaLimiterFactory.quota_limiters(config)
    assert limiters == []


def test_quota_limiters_empty_limiters():
    """Test the quota limiters creating when no limiters are specified."""
    config = QuotaHandlersConfig()
    config.storage = PostgresConfig()
    config.limiters = LimitersConfig()
    limiters = QuotaLimiterFactory.quota_limiters(config)
    assert limiters == []


def test_quota_limiters_user_quota_limiter():
    """Test the quota limiters creating when one limiter is specified."""
    config = QuotaHandlersConfig()
    config.storage = PostgresConfig()
    config.limiters = LimitersConfig(
        [
            {
                "name": "foo",
                "type": "user_limiter",
                "initial_quota": 1000,
                "quota_increase": 10,
                "period": "5 days",
            }
        ]
    )
    # do not use connection to real PostgreSQL instance
    with patch("psycopg2.connect"):
        limiters = QuotaLimiterFactory.quota_limiters(config)
        assert len(limiters) == 1
        assert isinstance(limiters[0], UserQuotaLimiter)


def test_quota_limiters_cluster_quota_limiter():
    """Test the quota limiters creating when one limiter is specified."""
    config = QuotaHandlersConfig()
    config.storage = PostgresConfig()
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
    # do not use connection to real PostgreSQL instance
    with patch("psycopg2.connect"):
        limiters = QuotaLimiterFactory.quota_limiters(config)
        assert len(limiters) == 1
        assert isinstance(limiters[0], ClusterQuotaLimiter)


def test_quota_limiters_two_limiters():
    """Test the quota limiters creating when two limiters are specified."""
    config = QuotaHandlersConfig()
    config.storage = PostgresConfig()
    config.limiters = LimitersConfig(
        [
            {
                "name": "foo",
                "type": "user_limiter",
                "initial_quota": 1000,
                "quota_increase": 10,
                "period": "5 days",
            },
            {
                "name": "bar",
                "type": "cluster_limiter",
                "initial_quota": 1000,
                "quota_increase": 10,
                "period": "5 days",
            },
        ]
    )
    # do not use connection to real PostgreSQL instance
    with patch("psycopg2.connect"):
        limiters = QuotaLimiterFactory.quota_limiters(config)
        assert len(limiters) == 2
        assert isinstance(limiters[0], UserQuotaLimiter)
        assert isinstance(limiters[1], ClusterQuotaLimiter)


def test_quota_limiters_unknown_limiter():
    """Test the quota limiters creating when the limiter type is unknown."""
    config = QuotaHandlersConfig()
    config.storage = PostgresConfig()
    config.limiters = LimitersConfig(
        [
            {
                "name": "foo",
                "type": "UNKNOWN",
                "initial_quota": 1000,
                "quota_increase": 10,
                "period": "5 days",
            }
        ]
    )
    # do not use connection to real PostgreSQL instance
    with patch("psycopg2.connect"):
        with pytest.raises(ValueError, match=r"Invalid limiter type: UNKNOWN\."):
            QuotaLimiterFactory.quota_limiters(config)
