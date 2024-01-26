"""Unit tests for configure_logging function."""

import logging

from ols.utils.logging import LoggingConfig, configure_logging


def test_configure_logging_shouldnt_log(caplog):
    """Test configure_logging function."""
    logging_config = LoggingConfig(
        {"app_log_level": "debug"},
    )

    configure_logging(logging_config)
    logger = logging.getLogger()
    logger.info("level too high")

    captured_out = caplog.text
    assert "" in captured_out


def test_configure_logging_should_log(caplog):
    """Test configure_logging function."""
    logging_config = LoggingConfig(
        {"app_log_level": "warning"},
    )

    configure_logging(logging_config)
    logger = logging.getLogger()
    logger.warning("tadada")

    captured_out = caplog.text
    assert "tadada" in captured_out
