"""Unit tests for configure_logging function."""

import logging

from ols.utils.logging import LoggingConfig, configure_logging


def test_configure_logging_shouldnt_log(caplog):
    """Test configure_logging function."""
    caplog.set_level(logging.INFO)
    logging_config = LoggingConfig(
        {"app_log_level": "info"},
    )

    configure_logging(logging_config)
    logger = logging.getLogger()
    logger.debug("level too low")

    captured_out = caplog.text
    assert captured_out == ""


def test_configure_logging_should_log(caplog):
    """Test configure_logging function."""
    caplog.set_level(logging.INFO)
    logging_config = LoggingConfig(
        {"app_log_level": "info"},
    )

    configure_logging(logging_config)
    logger = logging.getLogger()
    logger.info("tadada")

    captured_out = caplog.text
    assert "tadada" in captured_out
