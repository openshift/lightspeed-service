"""Unit tests for Logger class."""

import logging
import os
from unittest.mock import patch

from ols.utils.logger import Logger


@patch("dotenv.load_dotenv")
@patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"})
def test_logger_debug_level(mock_load_dotenv, capsys):
    """Test logger set with log level to DEBUG."""
    logger = Logger(logger_name="foo", log_level=logging.DEBUG)
    logger.logger.debug("Debug message")

    # check captured log output
    captured_out = capsys.readouterr().out
    assert "Debug message" in captured_out

    # error output should be empty
    captured_err = capsys.readouterr().err
    assert captured_err == ""


@patch("dotenv.load_dotenv")
@patch.dict(os.environ, {"LOG_LEVEL": "INFO"})
def test_logger_info_level(mock_load_dotenv, capsys):
    """Test logger set with log level to INFO."""
    logger = Logger(logger_name="foo", log_level=logging.INFO)
    logger.logger.debug("Debug message")

    # check captured log output
    # the log message should not be captured due to log level
    captured_out = capsys.readouterr().out
    assert captured_out == ""

    # error output should be empty as well
    captured_err = capsys.readouterr().err
    assert captured_err == ""


@patch("dotenv.load_dotenv")
@patch.dict(os.environ, {"LOG_LEVEL": "INFO"})
def test_logger_show_message_flag(mock_load_dotenv, capsys):
    """Test logger set with show_message flag."""
    logger = Logger(logger_name="foo", log_level=logging.INFO, show_message=True)
    logger.logger.debug("Debug message")

    # check captured log output
    # the log message should not be captured due to log level
    captured_out = capsys.readouterr().out
    assert "Set LOG_LEVEL environment variable (e.g., INFO, DEBUG)" in captured_out
    assert "Debug message" not in captured_out

    # error level should be empty
    captured_err = capsys.readouterr().err
    assert captured_err == ""


@patch("dotenv.load_dotenv")
@patch.dict(os.environ, {"LOG_LEVEL": "INFO"})
def test_logger_error_messages(mock_load_dotenv, capsys):
    """Test how logger log error messages."""
    logger = Logger(logger_name="foo", log_level=logging.INFO)
    logger.logger.error("Error message")

    # check captured log output
    # the log message should be captured
    captured_out = capsys.readouterr().out
    assert "ERROR: Error message" in captured_out

    # error level should be empty
    captured_err = capsys.readouterr().err
    assert captured_err == ""
