"""Simple wrapper around the Pyton logging function."""

import logging
import os
import sys

import dotenv


class Logger:
    """This class is a simple wrapper around the Python logging function.

    Usage:

        # Simple usage

        logger = Logger().logger

        logger.debug('Debug message')
        logger.info('Info message')
        logger.warning('Warning message')
        logger.error('Error message')
        logger.critical('Critical message')

        # If want to pass the name of the function generating the message
        # you may use the "inspect" library and generate the message as follows

        self.logger.debug(f"[{inspect.stack()[0][3]}] Message here.")

        # When using on a class that may already have another instance

        self.logger = logger if logger else Logger(show_message=False).logger

    """

    def __init__(
        self,
        logger_name: str = "default",
        log_level: str = logging.getLevelName(logging.INFO),
        show_message: bool = False,
    ):
        """Initializes the Logger instance.

        Args:
          logger_name: The name of the logger instance.
          log_level: The logging level for general logging verbosity.
          show_message: Whether to display a message about setting logging levels.

        Note:
        - The default values can be overridden using environment variable `LOG_LEVEL`.
        """
        msg = """
        #################################################################
        Set LOG_LEVEL environment variable (e.g., INFO, DEBUG) to control
        general logging verbosity.
        #################################################################
        """
        if show_message:
            print(msg)

        # Load the dotenv configuration in case config class has not been used
        dotenv.load_dotenv()

        self.logger_name = logger_name
        self.log_level = os.getenv("LOG_LEVEL", log_level)
        self.log_level_console = os.getenv("LOG_LEVEL_CONSOLE", self.log_level)

        self.set_handlers()

    def set_handlers(self) -> None:
        """Sets formatting, handler and logging levels."""
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
