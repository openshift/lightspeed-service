"""Unit tests for QuotaMetricsRepository."""

import datetime
from unittest.mock import MagicMock, call, patch
from typing import List

import pytest

from ols import config

# needs to be setup before imports that use authentication
config.ols_config.authentication_config.module = "k8s"

from ols.app.models.config import PostgresConfig
from ols.app.metrics.quota_metrics_repository import (
    QuotaMetricsRepository,
    PostgresQuotaMetricsRepository,
    QuotaRecord,
    TokenUsageRecord,
)


class TestQuotaRecord:
    """Test QuotaRecord data class."""

    def test_quota_record_creation(self):
        """Test QuotaRecord creation with all fields."""
        record = QuotaRecord(
            id="user123",
            subject="u",
            quota_limit=1000,
            available=750,
            updated_at=datetime.datetime(2024, 1, 1, 12, 0, 0),
        )
        assert record.id == "user123"
        assert record.subject == "u"
        assert record.quota_limit == 1000
        assert record.available == 750
        assert record.updated_at == datetime.datetime(2024, 1, 1, 12, 0, 0)

    def test_quota_utilization_percent(self):
        """Test quota utilization percentage calculation."""
        record = QuotaRecord(
            id="user123",
            subject="u",
            quota_limit=1000,
            available=250,
            updated_at=datetime.datetime(2024, 1, 1, 12, 0, 0),
        )
        assert record.utilization_percent == 75.0

    def test_quota_utilization_percent_zero_limit(self):
        """Test quota utilization when limit is zero."""
        record = QuotaRecord(
            id="user123",
            subject="u",
            quota_limit=0,
            available=0,
            updated_at=datetime.datetime(2024, 1, 1, 12, 0, 0),
        )
        assert record.utilization_percent == 0.0

    def test_quota_utilization_percent_negative_available(self):
        """Test quota utilization when available is negative (over quota)."""
        record = QuotaRecord(
            id="user123",
            subject="u",
            quota_limit=1000,
            available=-100,
            updated_at=datetime.datetime(2024, 1, 1, 12, 0, 0),
        )
        assert abs(record.utilization_percent - 110.0) < 0.001


class TestTokenUsageRecord:
    """Test TokenUsageRecord data class."""

    def test_token_usage_record_creation(self):
        """Test TokenUsageRecord creation with all fields."""
        record = TokenUsageRecord(
            user_id="user123",
            provider="openai",
            model="gpt-4",
            input_tokens=500,
            output_tokens=200,
            updated_at=datetime.datetime(2024, 1, 1, 12, 0, 0),
        )
        assert record.user_id == "user123"
        assert record.provider == "openai"
        assert record.model == "gpt-4"
        assert record.input_tokens == 500
        assert record.output_tokens == 200
        assert record.updated_at == datetime.datetime(2024, 1, 1, 12, 0, 0)

    def test_total_tokens(self):
        """Test total tokens calculation."""
        record = TokenUsageRecord(
            user_id="user123",
            provider="openai",
            model="gpt-4",
            input_tokens=500,
            output_tokens=200,
            updated_at=datetime.datetime(2024, 1, 1, 12, 0, 0),
        )
        assert record.total_tokens == 700


class TestPostgresQuotaMetricsRepository:
    """Test PostgresQuotaMetricsRepository implementation."""

    def test_init_storage_failure_detection(self):
        """Test exception handling for storage initialization."""
        exception_message = "Exception during PostgreSQL storage."

        with patch("psycopg2.connect") as mock_connect:
            mock_connect.side_effect = Exception(exception_message)

            config = PostgresConfig()
            with pytest.raises(Exception, match=exception_message):
                PostgresQuotaMetricsRepository(config)

    def test_get_quota_records_success(self):
        """Test successful retrieval of quota records."""
        quota_data = [
            ("user1", "u", 1000, 750, datetime.datetime(2024, 1, 1, 12, 0, 0)),
            ("user2", "u", 500, 200, datetime.datetime(2024, 1, 1, 13, 0, 0)),
            ("", "c", 10000, 8000, datetime.datetime(2024, 1, 1, 14, 0, 0)),
        ]

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = quota_data

        with patch("psycopg2.connect") as mock_connect:
            mock_connect.return_value.cursor.return_value.__enter__.return_value = (
                mock_cursor
            )

            config = PostgresConfig()
            repo = PostgresQuotaMetricsRepository(config)
            records = repo.get_quota_records()

            assert len(records) == 3
            assert records[0].id == "user1"
            assert records[0].subject == "u"
            assert records[0].quota_limit == 1000
            assert records[0].available == 750
            assert records[1].id == "user2"
            assert records[2].id == ""
            assert records[2].subject == "c"

    def test_get_quota_records_empty_result(self):
        """Test handling of empty quota records result."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []

        with patch("psycopg2.connect") as mock_connect:
            mock_connect.return_value.cursor.return_value.__enter__.return_value = (
                mock_cursor
            )

            config = PostgresConfig()
            repo = PostgresQuotaMetricsRepository(config)
            records = repo.get_quota_records()

            assert len(records) == 0

    def test_get_quota_records_database_error(self):
        """Test handling of database errors during quota record retrieval."""
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Database connection error")

        with patch("psycopg2.connect") as mock_connect:
            mock_connect.return_value.cursor.return_value.__enter__.return_value = (
                mock_cursor
            )

            config = PostgresConfig()
            repo = PostgresQuotaMetricsRepository(config)

            with pytest.raises(Exception, match="Database connection error"):
                repo.get_quota_records()

    def test_get_token_usage_records_success(self):
        """Test successful retrieval of token usage records."""
        token_data = [
            (
                "user1",
                "openai",
                "gpt-4",
                500,
                200,
                datetime.datetime(2024, 1, 1, 12, 0, 0),
            ),
            (
                "user2",
                "openai",
                "gpt-3.5",
                300,
                150,
                datetime.datetime(2024, 1, 1, 13, 0, 0),
            ),
        ]

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = token_data

        with patch("psycopg2.connect") as mock_connect:
            mock_connect.return_value.cursor.return_value.__enter__.return_value = (
                mock_cursor
            )

            config = PostgresConfig()
            repo = PostgresQuotaMetricsRepository(config)
            records = repo.get_token_usage_records()

            assert len(records) == 2
            assert records[0].user_id == "user1"
            assert records[0].provider == "openai"
            assert records[0].model == "gpt-4"
            assert records[0].input_tokens == 500
            assert records[0].output_tokens == 200
            assert records[1].user_id == "user2"

    def test_get_token_usage_records_empty_result(self):
        """Test handling of empty token usage records result."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []

        with patch("psycopg2.connect") as mock_connect:
            mock_connect.return_value.cursor.return_value.__enter__.return_value = (
                mock_cursor
            )

            config = PostgresConfig()
            repo = PostgresQuotaMetricsRepository(config)
            records = repo.get_token_usage_records()

            assert len(records) == 0

    def test_get_token_usage_records_database_error(self):
        """Test handling of database errors during token usage record retrieval."""
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Database connection error")

        with patch("psycopg2.connect") as mock_connect:
            mock_connect.return_value.cursor.return_value.__enter__.return_value = (
                mock_cursor
            )

            config = PostgresConfig()
            repo = PostgresQuotaMetricsRepository(config)

            with pytest.raises(Exception, match="Database connection error"):
                repo.get_token_usage_records()

    def test_health_check_success(self):
        """Test successful health check."""
        mock_cursor = MagicMock()

        with patch("psycopg2.connect") as mock_connect:
            mock_connect.return_value.cursor.return_value.__enter__.return_value = (
                mock_cursor
            )

            config = PostgresConfig()
            repo = PostgresQuotaMetricsRepository(config)
            is_healthy = repo.health_check()

            assert is_healthy is True
            mock_cursor.execute.assert_called_with("SELECT 1")

    def test_health_check_failure(self):
        """Test health check failure."""
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Connection failed")

        with patch("psycopg2.connect") as mock_connect:
            mock_connect.return_value.cursor.return_value.__enter__.return_value = (
                mock_cursor
            )

            config = PostgresConfig()
            repo = PostgresQuotaMetricsRepository(config)
            is_healthy = repo.health_check()

            assert is_healthy is False

    def test_disconnected_repository_reconnects(self):
        """Test that disconnected repository reconnects automatically."""
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []

        with patch("psycopg2.connect") as mock_connect:
            mock_connect.return_value.cursor.return_value.__enter__.return_value = (
                mock_cursor
            )

            config = PostgresConfig()
            repo = PostgresQuotaMetricsRepository(config)

            # Simulate disconnection
            repo.connection = None

            # This should trigger reconnection
            records = repo.get_quota_records()

            assert len(records) == 0
            # Verify reconnection was called
            assert mock_connect.call_count >= 2
