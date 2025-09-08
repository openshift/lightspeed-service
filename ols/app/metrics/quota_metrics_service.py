"""FastAPI dependency injection service for quota metrics."""

import logging
from typing import Optional

from ols.app.models.config import PostgresConfig
from ols.app.metrics.quota_metrics_collector import QuotaMetricsCollector
from ols.app.metrics.quota_metrics_repository import PostgresQuotaMetricsRepository

logger = logging.getLogger(__name__)

# Global collector instance for caching
_quota_metrics_collector: Optional[QuotaMetricsCollector] = None


def get_quota_metrics_collector(postgres_config: PostgresConfig) -> Optional[QuotaMetricsCollector]:
    """Get or create a quota metrics collector instance.
    
    This function serves as a FastAPI dependency that provides a quota metrics
    collector instance. It implements singleton pattern to avoid creating
    multiple database connections.
    
    Args:
        postgres_config: PostgreSQL configuration for database connection
        
    Returns:
        QuotaMetricsCollector instance or None if initialization fails
    """
    global _quota_metrics_collector
    
    if _quota_metrics_collector is not None:
        logger.debug("Returning cached quota metrics collector")
        return _quota_metrics_collector
    
    try:
        logger.info("Initializing quota metrics collector")
        repository = PostgresQuotaMetricsRepository(postgres_config)
        _quota_metrics_collector = QuotaMetricsCollector(repository)
        logger.info("Quota metrics collector initialized successfully")
        return _quota_metrics_collector
        
    except Exception as e:
        logger.error("Failed to initialize quota metrics collector: %s", e)
        # Re-raise to ensure the dependency system knows about the failure
        raise


def update_quota_metrics_on_request(collector: Optional[QuotaMetricsCollector]) -> None:
    """Update quota metrics when metrics endpoint is requested.
    
    This function can be called before serving metrics to ensure
    the latest quota data is included.
    
    Args:
        collector: The quota metrics collector instance
    """
    if collector is None:
        logger.warning("No quota metrics collector available, skipping update")
        return
    
    try:
        logger.debug("Updating quota metrics for endpoint request")
        collector.update_all_metrics()
        logger.debug("Quota metrics updated successfully")
        
    except Exception as e:
        logger.error("Failed to update quota metrics: %s", e)
        # Don't re-raise here to avoid breaking the metrics endpoint


def reset_quota_metrics_collector() -> None:
    """Reset the global quota metrics collector instance.
    
    This is primarily useful for testing to ensure clean state
    between test runs.
    """
    global _quota_metrics_collector
    logger.debug("Resetting quota metrics collector")
    _quota_metrics_collector = None