"""Logging utilities."""

import logging

from ols.app.models.config import LoggingConfig


def configure_logging(logging_config: LoggingConfig) -> None:
    """Configure application logging according to the configuration."""
    logging.basicConfig(
        level=logging_config.app_log_level,
        format="%(asctime)s [%(name)s:%(filename)s:%(lineno)d] %(levelname)s: %(message)s",
    )
    # manage logging levels for external libraries
    logging.getLogger("httpcore").setLevel(logging_config.library_log_level)
    logging.getLogger("httpx").setLevel(logging_config.library_log_level)
    logging.getLogger("urllib3").setLevel(logging_config.library_log_level)
