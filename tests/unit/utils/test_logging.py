"""Unit tests for configure_logging function."""

import logging

from ols.utils.logging import LoggingConfig, configure_logging


def test_configure_app_logging(caplog):
    """Test configure_logging function."""
    logging_config = LoggingConfig(
        {"app_log_level": "info"},
    )

    configure_logging(logging_config)
    logger = logging.getLogger("ols")
    logger.handlers = [caplog.handler]  # add caplog handler to logger
    logger.debug("debug msg")
    logger.info("info msg")

    captured_out = caplog.text
    assert "debug msg" not in captured_out
    assert "info msg" in captured_out


def test_configure_libs_logging(caplog):
    """Test configure_logging function for root/libs logger."""
    logging_config = LoggingConfig(
        {"lib_log_level": "info"},
    )

    configure_logging(logging_config)
    logger = logging.getLogger()
    logger.handlers = [caplog.handler]  # add caplog handler to logger
    logger.debug("debug msg")
    logger.info("info msg")

    captured_out = caplog.text
    assert "debug msg" not in captured_out
    assert "info msg" in captured_out
