"""Unit tests for Logger class."""

import logging

from ols.utils.logger import Logger


def test_logger_debug_level(capsys):
    """Test logger set with log level to DEBUG."""
    logger = Logger(logger_name="foo", log_level=logging.DEBUG)
    logger.logger.debug("Debug message")

    # check captured log output
    captured_out = capsys.readouterr().out
    assert "Debug message" in captured_out
    captured_err = capsys.readouterr().out
    assert captured_err == ""


def test_logger_info_level(capsys):
    """Test logger set with log level to INFO."""
    logger = Logger(logger_name="foo", log_level=logging.INFO)
    logger.logger.debug("Debug message")

    # check captured log output
    # the log message should not be captured due to log level
    captured_out = capsys.readouterr().out
    assert captured_out == ""
    captured_err = capsys.readouterr().out
    assert captured_err == ""


def test_logger_show_message_flag(capsys):
    """Test logger set with show_message flag."""
    logger = Logger(logger_name="foo", log_level=logging.INFO, show_message=True)
    logger.logger.debug("Debug message")

    # check captured log output
    # the log message should not be captured due to log level
    # but the message should be shown
    captured_out = capsys.readouterr().out
    assert "Set LOG_LEVEL environment variable (e.g., INFO, DEBUG)" in captured_out
    captured_err = capsys.readouterr().out
    assert captured_err == ""
