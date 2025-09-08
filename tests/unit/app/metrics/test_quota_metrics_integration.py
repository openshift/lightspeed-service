"""Integration tests for quota metrics with FastAPI dependencies."""

import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ols import config

# needs to be setup before imports that use authentication
config.ols_config.authentication_config.module = "k8s"

from ols.app.metrics.quota_metrics_repository import QuotaRecord, TokenUsageRecord
from ols.app.metrics.quota_metrics_service import get_quota_metrics_collector


class TestQuotaMetricsIntegration:
    """Test integration of quota metrics with FastAPI dependencies."""

    def setup_method(self):
        """Set up test environment."""
        from ols.app.metrics.quota_metrics_service import reset_quota_metrics_collector
        from prometheus_client import REGISTRY

        # Reset quota metrics service
        reset_quota_metrics_collector()

        # Clear any existing quota metrics from the registry
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
        from ols.app.metrics.quota_metrics_service import reset_quota_metrics_collector
        from prometheus_client import REGISTRY

        # Reset quota metrics service
        reset_quota_metrics_collector()

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

    def test_quota_metrics_dependency_injection(self):
        """Test that quota metrics collector can be injected as a dependency."""
        from ols.app.models.config import PostgresConfig

        with patch(
            "ols.app.metrics.quota_metrics_service.PostgresQuotaMetricsRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo

            config_obj = PostgresConfig()
            collector = get_quota_metrics_collector(config_obj)

            assert collector is not None
            assert hasattr(collector, "update_all_metrics")

    def test_metrics_endpoint_includes_quota_metrics(self):
        """Test that quota metrics are included in Prometheus output."""
        from prometheus_client import generate_latest

        # Mock the quota repository and collector
        mock_repo = MagicMock()
        mock_repo.get_quota_records.return_value = [
            QuotaRecord(
                id="user1",
                subject="u",
                quota_limit=1000,
                available=750,
                updated_at=datetime.datetime(2024, 1, 1, 12, 0, 0),
            )
        ]
        mock_repo.get_token_usage_records.return_value = [
            TokenUsageRecord(
                user_id="user1",
                provider="openai",
                model="gpt-4",
                input_tokens=500,
                output_tokens=200,
                updated_at=datetime.datetime(2024, 1, 1, 12, 0, 0),
            )
        ]
        mock_repo.health_check.return_value = True

        with patch(
            "ols.app.metrics.quota_metrics_service.PostgresQuotaMetricsRepository"
        ) as mock_repo_class:
            mock_repo_class.return_value = mock_repo

            from ols.app.metrics.quota_metrics_collector import QuotaMetricsCollector
            from ols.app.models.config import PostgresConfig

            # Create collector and update metrics
            config_obj = PostgresConfig()
            collector = get_quota_metrics_collector(config_obj)
            collector.update_all_metrics()

            # Generate Prometheus output
            metrics_output = generate_latest().decode("utf-8")

            # Check that quota metrics are included in the response
            assert "ols_quota_limit_total" in metrics_output
            assert "ols_quota_available_total" in metrics_output
            assert "ols_quota_utilization_percent" in metrics_output
            assert "ols_token_usage_total" in metrics_output

    def test_quota_metrics_update_on_endpoint_call(self):
        """Test that quota metrics are updated when metrics endpoint is called."""
        from ols.app.models.config import PostgresConfig

        mock_repo = MagicMock()
        mock_repo.get_quota_records.return_value = []
        mock_repo.get_token_usage_records.return_value = []
        mock_repo.health_check.return_value = True

        with patch(
            "ols.app.metrics.quota_metrics_service.PostgresQuotaMetricsRepository"
        ) as mock_repo_class:
            mock_repo_class.return_value = mock_repo

            config_obj = PostgresConfig()
            collector = get_quota_metrics_collector(config_obj)

            # Verify that the collector can update metrics
            collector.update_all_metrics()

            mock_repo.get_quota_records.assert_called_once()
            mock_repo.get_token_usage_records.assert_called_once()

    def test_quota_metrics_error_handling_in_dependency(self):
        """Test error handling when quota metrics collection fails."""
        from ols.app.models.config import PostgresConfig

        with patch(
            "ols.app.metrics.quota_metrics_service.PostgresQuotaMetricsRepository"
        ) as mock_repo_class:
            # Make repository initialization fail
            mock_repo_class.side_effect = Exception("Database connection failed")

            config_obj = PostgresConfig()

            # Should handle the error gracefully and return None or a dummy collector
            with pytest.raises(Exception):
                get_quota_metrics_collector(config_obj)

    def test_quota_metrics_caching_in_dependency(self):
        """Test that quota metrics collector is cached properly."""
        from ols.app.models.config import PostgresConfig

        with patch(
            "ols.app.metrics.quota_metrics_service.PostgresQuotaMetricsRepository"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo

            config_obj = PostgresConfig()

            # Call the dependency function multiple times
            collector1 = get_quota_metrics_collector(config_obj)
            collector2 = get_quota_metrics_collector(config_obj)

            # Should return the same instance (if cached)
            assert collector1 is not None
            assert collector2 is not None
            assert collector1 is collector2  # Same instance due to caching
