import logging, sys, os

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
    def __init__(self,
                 logger_name="default",
                 log_level=logging.INFO, 
                 logfile="logs/ols.logs",
                 show_message=False):
        msg=f"""
        ############################################################################
        Set LOG_LEVEL or LOG_LEVEL_CONSOLE environment variable (e.g., INFO, DEBUG) 
        to control general logging verbosity or console specific logging level
        ############################################################################
        """
        if show_message: print(msg)

        self.logger_name=logger_name
        self.log_level=os.getenv("LOG_LEVEL", log_level)
        self.log_level_console=os.getenv("LOG_LEVEL_CONSOLE", self.log_level)
        self.logfile=logfile
        self.logfile_maxSize=1048576*100
        self.logfile_backupCount=3

        self.set_handlers()
        
    def set_handlers(self):
        """
        A simple logger function that logs messages at a specified level.

        :param level:   The logging level (e.g. 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        :param message: The message to log
        """
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(self.log_level)
        
        formatter = logging.Formatter("%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s")

        # console logging handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level_console)
        console_handler.setStream(sys.stdout)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)

        # file logging handler (if not disabled)
        if self.logfile is not None:
            file_handler = logging.handlers.RotatingFileHandler(
                                                                self.logfile, 
                                                                maxBytes=self.logfile_maxSize, 
                                                                backupCount=self.logfile_backupCount)
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
