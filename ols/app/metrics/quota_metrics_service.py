"""FastAPI dependency injection service for quota metrics."""

import logging
from typing import Optional

from ols.app.metrics.quota_metrics_collector import QuotaMetricsCollector
from ols.app.metrics.quota_metrics_repository import PostgresQuotaMetricsRepository
from ols.app.models.config import PostgresConfig

logger = logging.getLogger(__name__)


class QuotaMetricsService:
    """Singleton service for managing quota metrics collector instances."""

    _instance: Optional["QuotaMetricsService"] = None
    _collector: Optional[QuotaMetricsCollector] = None

    def __new__(cls) -> "QuotaMetricsService":
        """Create a new instance using singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_collector(
        self, postgres_config: PostgresConfig
    ) -> Optional[QuotaMetricsCollector]:
        """Get or create a quota metrics collector instance.

        Args:
            postgres_config: PostgreSQL configuration for database connection

        Returns:
            QuotaMetricsCollector instance or None if initialization fails
        """
        if self._collector is not None:
            logger.debug("Returning cached quota metrics collector")
            return self._collector

        try:
            logger.info("Initializing quota metrics collector")
            repository = PostgresQuotaMetricsRepository(postgres_config)
            self._collector = QuotaMetricsCollector(repository)
            logger.info("Quota metrics collector initialized successfully")
            return self._collector

        except Exception as e:
            logger.error("Failed to initialize quota metrics collector: %s", e)
            # Re-raise to ensure the dependency system knows about the failure
            raise

    def reset(self) -> None:
        """Reset the collector instance."""
        logger.debug("Resetting quota metrics collector")
        self._collector = None


# Module-level service instance
_service = QuotaMetricsService()


def get_quota_metrics_collector(
    postgres_config: PostgresConfig,
) -> Optional[QuotaMetricsCollector]:
    """Get or create a quota metrics collector instance.

    This function serves as a FastAPI dependency that provides a quota metrics
    collector instance. It implements singleton pattern to avoid creating
    multiple database connections.

    Args:
        postgres_config: PostgreSQL configuration for database connection

    Returns:
        QuotaMetricsCollector instance or None if initialization fails
    """
    return _service.get_collector(postgres_config)


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
    """Reset the quota metrics collector instance.

    This is primarily useful for testing to ensure clean state
    between test runs.
    """
    _service.reset()
