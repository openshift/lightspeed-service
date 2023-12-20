import logging
import os
import sys
from logging.handlers import RotatingFileHandler

import dotenv


class Logger:
    """
    This class is a simple wrapper around the Python logging function

    Usage:

        # Simple usage

        logger = Logger().logger

        logger.debug('Debug message')
        logger.info('Info message')
        logger.warning('Warning message')
        logger.error('Error message')
        logger.critical('Critical message')

        # To disable logging to file set logfile to None

        logger = Logger(logfile=None).logger

        # If want to pass the name of the function generating the message
        # you may use the "inspect" library and generate the message as follows

        self.logger.debug(f"[{inspect.stack()[0][3]}] Message here.")

        # When using on a class that may already have another instance

        self.logger = logger if logger else Logger(show_message=False).logger

    """

    def __init__(
        self,
        logger_name="default",
        log_level=logging.INFO,
        logfile=None,
        show_message=False,
    ):
        """
        Initializes the Logger instance.

        Args:
        - `logger_name` (str): The name of the logger instance.
        - `log_level` (int): The logging level for general logging verbosity.
        - `logfile` (str): The path to the log file. Set to `None` to disable file logging.
        - `show_message` (bool): Whether to display a message about setting logging levels.

        Note:
        - The default values can be overridden using environment variables `LOG_LEVEL`
          and `LOG_LEVEL_CONSOLE`.
        - To set logfile name set `LOG_FILE_NAME`
        - To override logfile maximum size set `LOG_FILE_SIZE`
        """
        msg = """
        ############################################################################
        Set LOG_LEVEL or LOG_LEVEL_CONSOLE environment variable (e.g., INFO, DEBUG)
        to control general logging verbosity or console specific logging level
        ############################################################################
        """
        if show_message:
            print(msg)

        # Load the dotenv configuration in case config class has not been used
        dotenv.load_dotenv()

        self.logger_name = logger_name
        self.log_level = os.getenv("LOG_LEVEL", log_level)
        self.log_level_console = os.getenv("LOG_LEVEL_CONSOLE", self.log_level)
        _logfile = os.getenv("LOG_FILE_NAME")
        self.logfile = _logfile if _logfile else logfile
        self.logfile_maxSize = int(os.getenv("LOG_FILE_SIZE", (1048576 * 100)))
        self.logfile_backupCount = 3

        self.set_handlers()

    def set_handlers(self):
        """
        A simple logger function that logs messages at a specified level.

        :param level:   The logging level (e.g. 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        :param message: The message to log
        """
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(self.log_level)

        formatter = logging.Formatter(
            "%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s"
        )

        # console logging handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level_console)
        console_handler.setStream(sys.stdout)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)

        # file logging handler (if not disabled)
        if self.logfile is not None:
            file_handler = RotatingFileHandler(
                self.logfile,
                maxBytes=self.logfile_maxSize,
                backupCount=self.logfile_backupCount,
            )
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
