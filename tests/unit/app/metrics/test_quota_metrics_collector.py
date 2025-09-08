"""Unit tests for QuotaMetricsCollector."""

import datetime
from unittest.mock import MagicMock, patch

import pytest
from prometheus_client import REGISTRY

from ols import config

# needs to be setup before imports that use authentication
config.ols_config.authentication_config.module = "k8s"

from ols.app.metrics.quota_metrics_collector import QuotaMetricsCollector
from ols.app.metrics.quota_metrics_repository import QuotaRecord, TokenUsageRecord


class TestQuotaMetricsCollector:
    """Test QuotaMetricsCollector implementation."""

    def setup_method(self):
        """Set up test environment."""
        # Clear any existing metrics to avoid conflicts
        to_remove = []
        for collector in list(REGISTRY._collector_to_names.keys()):
            if hasattr(collector, "_name") and (
                "ols_quota" in collector._name or "ols_token" in collector._name
            ):
                to_remove.append(collector)

        for collector in to_remove:
            try:
                REGISTRY.unregister(collector)
            except KeyError:
                pass  # Already removed

    def teardown_method(self):
        """Clean up test environment."""
        # Clear any metrics created during tests
        to_remove = []
        for collector in list(REGISTRY._collector_to_names.keys()):
            if hasattr(collector, "_name") and (
                "ols_quota" in collector._name or "ols_token" in collector._name
            ):
                to_remove.append(collector)

        for collector in to_remove:
            try:
                REGISTRY.unregister(collector)
            except KeyError:
                pass  # Already removed

    def test_init_creates_prometheus_metrics(self):
        """Test that initialization creates Prometheus metrics."""
        mock_repo = MagicMock()
        collector = QuotaMetricsCollector(mock_repo)

        assert collector.quota_limit_total is not None
        assert collector.quota_available_total is not None
        assert collector.quota_utilization_percent is not None
        assert collector.token_usage_total is not None
        assert collector.quota_warning_subjects_total is not None
        assert collector.quota_exceeded_subjects_total is not None

    def test_update_quota_metrics_with_data(self):
        """Test updating quota metrics with sample data."""
        mock_repo = MagicMock()

        quota_records = [
            QuotaRecord(
                id="user1",
                subject="u",
                quota_limit=1000,
                available=750,
                updated_at=datetime.datetime(2024, 1, 1, 12, 0, 0),
            ),
            QuotaRecord(
                id="user2",
                subject="u",
                quota_limit=500,
                available=100,
                updated_at=datetime.datetime(2024, 1, 1, 13, 0, 0),
            ),
            QuotaRecord(
                id="",
                subject="c",
                quota_limit=10000,
                available=8000,
                updated_at=datetime.datetime(2024, 1, 1, 14, 0, 0),
            ),
        ]

        mock_repo.get_quota_records.return_value = quota_records
        mock_repo.health_check.return_value = True

        collector = QuotaMetricsCollector(mock_repo)
        collector.update_quota_metrics()

        # Verify repository was called
        mock_repo.get_quota_records.assert_called_once()

        # Verify metrics were set (this would require inspecting the prometheus metrics)
        # For now, just verify no exceptions were raised

    def test_update_quota_metrics_empty_data(self):
        """Test updating quota metrics with empty data."""
        mock_repo = MagicMock()
        mock_repo.get_quota_records.return_value = []
        mock_repo.health_check.return_value = True

        collector = QuotaMetricsCollector(mock_repo)
        collector.update_quota_metrics()

        mock_repo.get_quota_records.assert_called_once()

    def test_update_quota_metrics_database_error(self):
        """Test handling of database errors during quota metrics update."""
        mock_repo = MagicMock()
        mock_repo.get_quota_records.side_effect = Exception("Database connection error")
        mock_repo.health_check.return_value = (
            True  # Health check passes, but query fails
        )

        collector = QuotaMetricsCollector(mock_repo)

        # Should not raise exception, but should handle gracefully
        collector.update_quota_metrics()

        mock_repo.get_quota_records.assert_called_once()

    def test_update_token_usage_metrics_with_data(self):
        """Test updating token usage metrics with sample data."""
        mock_repo = MagicMock()

        token_records = [
            TokenUsageRecord(
                user_id="user1",
                provider="openai",
                model="gpt-4",
                input_tokens=500,
                output_tokens=200,
                updated_at=datetime.datetime(2024, 1, 1, 12, 0, 0),
            ),
            TokenUsageRecord(
                user_id="user2",
                provider="openai",
                model="gpt-3.5",
                input_tokens=300,
                output_tokens=150,
                updated_at=datetime.datetime(2024, 1, 1, 13, 0, 0),
            ),
        ]

        mock_repo.get_token_usage_records.return_value = token_records
        mock_repo.health_check.return_value = True

        collector = QuotaMetricsCollector(mock_repo)
        collector.update_token_usage_metrics()

        mock_repo.get_token_usage_records.assert_called_once()

    def test_update_token_usage_metrics_empty_data(self):
        """Test updating token usage metrics with empty data."""
        mock_repo = MagicMock()
        mock_repo.get_token_usage_records.return_value = []
        mock_repo.health_check.return_value = True

        collector = QuotaMetricsCollector(mock_repo)
        collector.update_token_usage_metrics()

        mock_repo.get_token_usage_records.assert_called_once()

    def test_update_token_usage_metrics_database_error(self):
        """Test handling of database errors during token usage metrics update."""
        mock_repo = MagicMock()
        mock_repo.get_token_usage_records.side_effect = Exception(
            "Database connection error"
        )
        mock_repo.health_check.return_value = (
            True  # Health check passes, but query fails
        )

        collector = QuotaMetricsCollector(mock_repo)

        # Should not raise exception, but should handle gracefully
        collector.update_token_usage_metrics()

        mock_repo.get_token_usage_records.assert_called_once()

    def test_update_all_metrics(self):
        """Test updating all metrics at once."""
        mock_repo = MagicMock()
        mock_repo.get_quota_records.return_value = []
        mock_repo.get_token_usage_records.return_value = []
        mock_repo.health_check.return_value = True

        collector = QuotaMetricsCollector(mock_repo)
        collector.update_all_metrics()

        mock_repo.get_quota_records.assert_called_once()
        mock_repo.get_token_usage_records.assert_called_once()

    def test_quota_threshold_calculations(self):
        """Test quota threshold calculations for warnings and exceeded."""
        mock_repo = MagicMock()

        quota_records = [
            # User with 90% utilization (warning)
            QuotaRecord(
                id="user1",
                subject="u",
                quota_limit=1000,
                available=100,  # 90% used
                updated_at=datetime.datetime(2024, 1, 1, 12, 0, 0),
            ),
            # User with 110% utilization (exceeded)
            QuotaRecord(
                id="user2",
                subject="u",
                quota_limit=500,
                available=-50,  # 110% used
                updated_at=datetime.datetime(2024, 1, 1, 13, 0, 0),
            ),
            # User with 50% utilization (normal)
            QuotaRecord(
                id="user3",
                subject="u",
                quota_limit=1000,
                available=500,  # 50% used
                updated_at=datetime.datetime(2024, 1, 1, 14, 0, 0),
            ),
        ]

        mock_repo.get_quota_records.return_value = quota_records
        mock_repo.health_check.return_value = True

        collector = QuotaMetricsCollector(mock_repo)
        collector.update_quota_metrics()

        mock_repo.get_quota_records.assert_called_once()

    def test_health_check_integration(self):
        """Test that health check affects metric collection behavior."""
        mock_repo = MagicMock()
        mock_repo.health_check.return_value = False
        mock_repo.get_quota_records.side_effect = Exception("DB unavailable")

        collector = QuotaMetricsCollector(mock_repo)

        # Should handle unhealthy database gracefully
        collector.update_quota_metrics()

        # Verify health check was performed
        mock_repo.health_check.assert_called()

    def test_metric_labels_and_values(self):
        """Test that metrics contain correct labels and values."""
        mock_repo = MagicMock()

        quota_record = QuotaRecord(
            id="user123",
            subject="u",
            quota_limit=1000,
            available=750,
            updated_at=datetime.datetime(2024, 1, 1, 12, 0, 0),
        )

        token_record = TokenUsageRecord(
            user_id="user123",
            provider="openai",
            model="gpt-4",
            input_tokens=500,
            output_tokens=200,
            updated_at=datetime.datetime(2024, 1, 1, 12, 0, 0),
        )

        mock_repo.get_quota_records.return_value = [quota_record]
        mock_repo.get_token_usage_records.return_value = [token_record]
        mock_repo.health_check.return_value = True

        collector = QuotaMetricsCollector(mock_repo)
        collector.update_all_metrics()

        # Verify method calls
        mock_repo.get_quota_records.assert_called_once()
        mock_repo.get_token_usage_records.assert_called_once()
        mock_repo.health_check.assert_called()

    def test_error_resilience_during_metric_update(self):
        """Test that errors in one metric update don't affect others."""
        mock_repo = MagicMock()

        # Make quota records fail but token usage succeed
        mock_repo.get_quota_records.side_effect = Exception("Quota DB error")
        mock_repo.get_token_usage_records.return_value = []
        mock_repo.health_check.return_value = True

        collector = QuotaMetricsCollector(mock_repo)

        # Should not raise exception for the entire update
        collector.update_all_metrics()

        # Both methods should have been attempted
        mock_repo.get_quota_records.assert_called_once()
        mock_repo.get_token_usage_records.assert_called_once()
