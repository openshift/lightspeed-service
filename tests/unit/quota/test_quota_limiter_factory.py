"""Unit tests for quota limiter factory class."""

from unittest.mock import patch

import pytest

from ols.app.models.config import LimitersConfig, PostgresConfig, QuotaLimiterConfig
from ols.src.quota.cluster_quota_limiter import ClusterQuotaLimiter
from ols.src.quota.quota_limiter_factory import QuotaLimiterFactory
from ols.src.quota.user_quota_limiter import UserQuotaLimiter


def test_quota_limiters_no_storage():
    """Test the quota limiters creating when no storage is configured."""
    config = QuotaLimiterConfig()
    config.storage = None
    limiters = QuotaLimiterFactory.quota_limiters(config)
    assert limiters == []


def test_quota_limiters_no_limiters():
    """Test the quota limiters creating when no limiters are specified."""
    config = QuotaLimiterConfig()
    config.storage = PostgresConfig()
    config.limiters = None
    limiters = QuotaLimiterFactory.quota_limiters(config)
    assert limiters == []


def test_quota_limiters_empty_limiters():
    """Test the quota limiters creating when no limiters are specified."""
    config = QuotaLimiterConfig()
    config.storage = PostgresConfig()
    config.limiters = LimitersConfig()
    limiters = QuotaLimiterFactory.quota_limiters(config)
    assert limiters == []


@patch("psycopg2.connect")
def test_quota_limiters_user_quota_limiter(postgres_mock):
    """Test the quota limiters creating when one limiter is specified."""
    config = QuotaLimiterConfig()
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
    limiters = QuotaLimiterFactory.quota_limiters(config)
    assert len(limiters) == 1
    assert isinstance(limiters[0], UserQuotaLimiter)


@patch("psycopg2.connect")
def test_quota_limiters_cluster_quota_limiter(postgres_mock):
    """Test the quota limiters creating when one limiter is specified."""
    config = QuotaLimiterConfig()
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
    limiters = QuotaLimiterFactory.quota_limiters(config)
    assert len(limiters) == 1
    assert isinstance(limiters[0], ClusterQuotaLimiter)


@patch("psycopg2.connect")
def test_quota_limiters_two_limiters(postgres_mock):
    """Test the quota limiters creating when two limiters are specified."""
    config = QuotaLimiterConfig()
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
    limiters = QuotaLimiterFactory.quota_limiters(config)
    assert len(limiters) == 2
    assert isinstance(limiters[0], UserQuotaLimiter)
    assert isinstance(limiters[1], ClusterQuotaLimiter)


@patch("psycopg2.connect")
def test_quota_limiters_unknown_limiter(postgres_mock):
    """Test the quota limiters creating when the limiter type is unknown."""
    config = QuotaLimiterConfig()
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
    with pytest.raises(ValueError, match="Invalid limiter type: UNKNOWN."):
        QuotaLimiterFactory.quota_limiters(config)
