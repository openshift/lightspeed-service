import logging
import os
from tempfile import gettempdir

from utils.logger import Logger

# log file should be stored in temporary directory
log_file_name = os.path.join(gettempdir(), "test.log")


def remove_logfile():
    """Remove logfile, if exists."""
    if os.path.exists(log_file_name):
        os.remove(log_file_name)


def setup_function(function):
    """Setup test environment for this module."""
    remove_logfile()


def teardown_function(function):
    """Teardown test environment for this module."""
    remove_logfile()


def test_logger_debug_level(capsys):
    """Test logger set with log level to DEBUG."""
    logger = Logger(logger_name="foo", log_level=logging.DEBUG, logfile=None)
    logger.logger.debug("Debug message")

    # check captured log output
    captured_out = capsys.readouterr().out
    assert "Debug message" in captured_out
    captured_err = capsys.readouterr().out
    assert captured_err == ""


def test_logger_info_level(capsys):
    """Test logger set with log level to INFO."""
    logger = Logger(logger_name="foo", log_level=logging.INFO, logfile=None)
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
    assert "Set LOG_LEVEL or LOG_LEVEL_CONSOLE environment variabl" in captured_out
    captured_err = capsys.readouterr().out
    assert captured_err == ""


def test_logging_to_file():
    """Test logging to file."""
    logger = Logger(logger_name="foo", log_level=logging.INFO, logfile=log_file_name)
    logger.logger.info("Info message")

    # need to flush everything onto the log
    logging.shutdown()

    # check if the message has been logged properly
    with open(log_file_name, "r") as fin:
        content = fin.read()
        assert "Info message" in content
