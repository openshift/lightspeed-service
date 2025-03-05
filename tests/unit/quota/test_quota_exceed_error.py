"""Unit tests for QuotaExceedError class."""

import pytest

from ols.src.quota.quota_exceed_error import QuotaExceedError


def test_quota_exceed_error_constructor():
    """Test the QuotaExceedError constructor."""
    expected = "User 1234 has 100 tokens, but 1000 tokens are needed"
    with pytest.raises(QuotaExceedError, match=expected):
        raise QuotaExceedError("1234", "u", 100, 1000)

    expected = "Cluster has 100 tokens, but 1000 tokens are needed"
    with pytest.raises(QuotaExceedError, match=expected):
        raise QuotaExceedError("", "c", 100, 1000)

    expected = "Unknown subject 1234 has 100 tokens, but 1000 tokens are needed"
    with pytest.raises(QuotaExceedError, match=expected):
        raise QuotaExceedError("1234", "?", 100, 1000)

    expected = "User 1234 has no available tokens"
    with pytest.raises(QuotaExceedError, match=expected):
        raise QuotaExceedError("1234", "u", 0, 0)

    expected = "Cluster has no available tokens"
    with pytest.raises(QuotaExceedError, match=expected):
        raise QuotaExceedError("", "c", 0, 0)

    expected = "Unknown subject 1234 has no available tokens"
    with pytest.raises(QuotaExceedError, match=expected):
        raise QuotaExceedError("1234", "?", 0, 0)
