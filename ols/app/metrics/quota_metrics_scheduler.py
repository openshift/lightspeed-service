"""Background scheduler for quota metrics collection."""

import logging
import time
from threading import Thread
from typing import Optional

from ols.app.models.config import QuotaHandlersConfig
from ols.app.metrics.quota_metrics_service import get_quota_metrics_collector
from ols.utils.config import AppConfig

logger = logging.getLogger(__name__)

# Default update interval in seconds (5 minutes)
DEFAULT_UPDATE_INTERVAL = 300


def quota_metrics_scheduler(config: Optional[QuotaHandlersConfig]) -> bool:
    """Background task to periodically update quota metrics.

    This function runs in an infinite loop, updating quota metrics at regular
    intervals. It's designed to be run in a separate daemon thread.

    Args:
        config: Quota handlers configuration containing storage and scheduler settings

    Returns:
        False if configuration is invalid or initialization fails,
        otherwise runs indefinitely
    """
    if config is None:
        logger.warning(
            "Quota handlers not configured, skipping quota metrics scheduler"
        )
        return False

    if config.storage is None:
        logger.warning(
            "Storage for quota handlers not configured, skipping quota metrics scheduler"
        )
        return False

    # Determine update interval
    update_interval = DEFAULT_UPDATE_INTERVAL
    if config.scheduler is not None and config.scheduler.period is not None:
        update_interval = config.scheduler.period

    logger.info(
        "Starting quota metrics scheduler with %d second interval", update_interval
    )

    # Initialize quota metrics collector
    try:
        collector = get_quota_metrics_collector(config.storage)
        if collector is None:
            logger.error("Failed to initialize quota metrics collector")
            return False
    except Exception as e:
        logger.error("Error initializing quota metrics collector: %s", e)
        return False

    # Main scheduler loop
    while True:
        try:
            logger.debug("Updating quota metrics")
            collector.update_all_metrics()
            logger.debug("Quota metrics updated successfully")

        except Exception as e:
            logger.error("Error updating quota metrics: %s", e)
            # Continue running even if update fails

        try:
            time.sleep(update_interval)
        except KeyboardInterrupt:
            logger.info("Quota metrics scheduler interrupted, stopping")
            break

    return True


def start_quota_metrics_scheduler(config: AppConfig) -> None:
    """Start quota metrics scheduler in a separate daemon thread.

    Args:
        config: Application configuration containing quota handler settings
    """
    logger.info("Starting quota metrics scheduler thread")

    thread = Thread(
        target=quota_metrics_scheduler,
        daemon=True,
        args=(config.ols_config.quota_handlers,),
    )
    thread.start()

    logger.info("Quota metrics scheduler thread started")
