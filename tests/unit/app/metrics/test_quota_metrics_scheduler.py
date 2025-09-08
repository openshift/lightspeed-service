"""Unit tests for quota metrics scheduler."""

import time
from threading import Thread
from unittest.mock import MagicMock, patch

import pytest

from ols import config

# needs to be setup before imports that use authentication
config.ols_config.authentication_config.module = "k8s"

from ols.app.metrics.quota_metrics_scheduler import (
    quota_metrics_scheduler,
    start_quota_metrics_scheduler,
)
from ols.utils.config import AppConfig


class TestQuotaMetricsScheduler:
    """Test quota metrics scheduler functionality."""

    def test_quota_metrics_scheduler_with_no_config(self):
        """Test scheduler behavior when quota handlers are not configured."""
        result = quota_metrics_scheduler(None)
        assert result is False

    def test_quota_metrics_scheduler_with_no_storage(self):
        """Test scheduler behavior when storage is not configured."""
        from ols.app.models.config import QuotaHandlersConfig
        
        config_obj = QuotaHandlersConfig()
        config_obj.storage = None
        
        result = quota_metrics_scheduler(config_obj)
        assert result is False

    def test_quota_metrics_scheduler_with_valid_config(self):
        """Test scheduler with valid configuration."""
        from ols.app.models.config import QuotaHandlersConfig, PostgresConfig
        
        config_obj = QuotaHandlersConfig()
        config_obj.storage = PostgresConfig()
        
        with patch("ols.app.metrics.quota_metrics_scheduler.get_quota_metrics_collector") as mock_get_collector:
            mock_collector = MagicMock()
            mock_get_collector.return_value = mock_collector
            
            with patch("time.sleep") as mock_sleep:
                # Make sleep raise an exception to exit the loop
                mock_sleep.side_effect = KeyboardInterrupt("Test exit")
                
                try:
                    quota_metrics_scheduler(config_obj)
                except KeyboardInterrupt:
                    pass  # Expected to exit this way
                
                # Verify collector was called
                mock_get_collector.assert_called_once()
                mock_collector.update_all_metrics.assert_called()

    def test_quota_metrics_scheduler_with_collector_error(self):
        """Test scheduler behavior when collector initialization fails."""
        from ols.app.models.config import QuotaHandlersConfig, PostgresConfig
        
        config_obj = QuotaHandlersConfig()
        config_obj.storage = PostgresConfig()
        
        with patch("ols.app.metrics.quota_metrics_scheduler.get_quota_metrics_collector") as mock_get_collector:
            mock_get_collector.side_effect = Exception("Database connection failed")
            
            result = quota_metrics_scheduler(config_obj)
            assert result is False

    def test_quota_metrics_scheduler_continues_on_update_error(self):
        """Test that scheduler continues running even if update fails."""
        from ols.app.models.config import QuotaHandlersConfig, PostgresConfig
        
        config_obj = QuotaHandlersConfig()
        config_obj.storage = PostgresConfig()
        
        with patch("ols.app.metrics.quota_metrics_scheduler.get_quota_metrics_collector") as mock_get_collector:
            mock_collector = MagicMock()
            mock_collector.update_all_metrics.side_effect = [
                Exception("First update failed"),
                None,  # Second update succeeds
            ]
            mock_get_collector.return_value = mock_collector
            
            with patch("time.sleep") as mock_sleep:
                # Make sleep raise an exception to exit the loop after second iteration
                mock_sleep.side_effect = [None, KeyboardInterrupt("Test exit")]
                
                try:
                    quota_metrics_scheduler(config_obj)
                except KeyboardInterrupt:
                    pass  # Expected to exit this way
                
                # Verify collector was called twice
                assert mock_collector.update_all_metrics.call_count == 2

    def test_quota_metrics_scheduler_sleep_interval(self):
        """Test that scheduler respects the configured sleep interval."""
        from ols.app.models.config import QuotaHandlersConfig, PostgresConfig, SchedulerConfig
        
        config_obj = QuotaHandlersConfig()
        config_obj.storage = PostgresConfig()
        config_obj.scheduler = SchedulerConfig(period=42)  # Custom period
        
        with patch("ols.app.metrics.quota_metrics_scheduler.get_quota_metrics_collector") as mock_get_collector:
            mock_collector = MagicMock()
            mock_get_collector.return_value = mock_collector
            
            with patch("time.sleep") as mock_sleep:
                mock_sleep.side_effect = KeyboardInterrupt("Test exit")
                
                try:
                    quota_metrics_scheduler(config_obj)
                except KeyboardInterrupt:
                    pass  # Expected to exit this way
                
                # Verify sleep was called with correct interval
                mock_sleep.assert_called_with(42)

    def test_start_quota_metrics_scheduler(self):
        """Test starting the quota metrics scheduler in a background thread."""
        from ols.app.models.config import QuotaHandlersConfig, PostgresConfig
        
        config_obj = AppConfig()
        # Set up quota handlers config
        config_obj.ols_config.quota_handlers = QuotaHandlersConfig()
        config_obj.ols_config.quota_handlers.storage = PostgresConfig()
        
        with patch("ols.app.metrics.quota_metrics_scheduler.Thread") as mock_thread_class:
            mock_thread = MagicMock()
            mock_thread_class.return_value = mock_thread
            
            start_quota_metrics_scheduler(config_obj)
            
            # Verify thread was created and started
            mock_thread_class.assert_called_once()
            call_args = mock_thread_class.call_args
            assert call_args[1]["daemon"] is True
            assert call_args[1]["target"] == quota_metrics_scheduler
            assert call_args[1]["args"] == (config_obj.ols_config.quota_handlers,)
            
            mock_thread.start.assert_called_once()

    def test_quota_metrics_scheduler_default_interval(self):
        """Test scheduler uses default interval when not configured."""
        from ols.app.models.config import QuotaHandlersConfig, PostgresConfig
        
        config_obj = QuotaHandlersConfig()
        config_obj.storage = PostgresConfig()
        config_obj.scheduler = None  # No scheduler config
        
        with patch("ols.app.metrics.quota_metrics_scheduler.get_quota_metrics_collector") as mock_get_collector:
            mock_collector = MagicMock()
            mock_get_collector.return_value = mock_collector
            
            with patch("time.sleep") as mock_sleep:
                mock_sleep.side_effect = KeyboardInterrupt("Test exit")
                
                try:
                    quota_metrics_scheduler(config_obj)
                except KeyboardInterrupt:
                    pass  # Expected to exit this way
                
                # Verify sleep was called with default interval (300 seconds)
                mock_sleep.assert_called_with(300)

    def test_quota_metrics_scheduler_thread_safety(self):
        """Test that scheduler can be safely interrupted."""
        from ols.app.models.config import QuotaHandlersConfig, PostgresConfig
        
        config_obj = QuotaHandlersConfig()
        config_obj.storage = PostgresConfig()
        
        with patch("ols.app.metrics.quota_metrics_scheduler.get_quota_metrics_collector") as mock_get_collector:
            mock_collector = MagicMock()
            mock_get_collector.return_value = mock_collector
            
            # Start scheduler in a real thread
            thread = Thread(
                target=quota_metrics_scheduler,
                daemon=True,
                args=(config_obj,)
            )
            thread.start()
            
            # Let it run briefly
            time.sleep(0.1)
            
            # Thread should be running
            assert thread.is_alive()
            
            # The daemon thread will be automatically cleaned up when test ends