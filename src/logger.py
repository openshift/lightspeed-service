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

        # If want to pass the name of the function generating the message
        # you may use the "inspect" library and generate the message as follows 

        self.logger.debug(f"[{inspect.stack()[0][3]}] Message here.")

        # When using on a class that may already have another instance

        self.logger = logger if logger else Logger(show_message=False).logger

    """
    def __init__(self, log_level=logging.INFO, 
                 logfile="logs/logs.txt",
                 show_message=False):
        msg=f"""
        ############################################################################
        Set LOG_LEVEL environment variable (e.g., INFO, DEBUG) to cocntrol verbosity
        ############################################################################
        """
        if show_message: print(msg)

        self.log_level=os.getenv("LOG_LEVEL", log_level)
        self.logfile=logfile
        self.set_handlers()
        
    def set_handlers(self):
        """
        A simple logger function that logs messages at a specified level.

        :param level:   The logging level (e.g. 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        :param message: The message to log
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.log_level)

        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

        # console logging handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setStream(sys.stdout)
        console_handler.setFormatter(formatter)

        # file logging handler
        file_handler = logging.FileHandler(self.logfile)
        file_handler.setLevel(logging.DEBUG)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
