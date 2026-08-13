
# logging at the top so it propogates
import logging
# from haggis.logs import *
def add_logging_level(level_name, level_num, method_name=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.
    """
    if not method_name:
        method_name = level_name.lower()

    if hasattr(logging, level_name):
        raise AttributeError(f'{level_name} already defined in logging module')
    if hasattr(logging, method_name):
        raise AttributeError(f'{method_name} already defined in logging module')

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def log_for_level(self, message, *args, **kwargs):
        if self.isEnabledFor(level_num):
            self._log(level_num, message, args, **kwargs, stacklevel=2)
    def log_to_root(message, *args, **kwargs):
        logging.log(level_num, message, *args, **kwargs, stacklevel=2)

    logging.addLevelName(level_num, level_name)
    setattr(logging, level_name, level_num)
    setattr(logging.getLoggerClass(), method_name, log_for_level)
    setattr(logging, method_name, log_to_root)



add_logging_level('INFOX', logging.INFO - 5)
add_logging_level('TRACE', logging.DEBUG - 5)

class TraceFilter(logging.Filter):
    def filter(self, record):
        if record.levelno == logging.getLevelName('TRACE'):
            record.funcName = record.funcName
        else:
            record.funcName = ''
        return True

# Function to set log level for all imported modules
def set_log_level(loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {loglevel}")

    # Define formatters
    trace_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s')
    default_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Get the root logger
    root_logger = logging.getLogger()
    
    # Clear existing handlers on root logger
    root_logger.handlers = []

    # Create handlers
    handler = logging.StreamHandler()
    handler.setLevel(numeric_level)

    # Apply the custom filter
    handler.addFilter(TraceFilter())

    # Apply the appropriate formatter
    if numeric_level == logging.TRACE:
        handler.setFormatter(trace_formatter)
    else:
        handler.setFormatter(default_formatter)


    # Configure the root logger
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(handler)

    # Set the log level and handler for all imported modules
    for module in logging.root.manager.loggerDict:
        mod_logger = logging.getLogger(module)
        mod_logger.setLevel(numeric_level)


# decorator from a discussion with claude AI on zotgpt
from functools import wraps
def log_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            logger.error(f"ValueError in {func.__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper

import sys
import inspect
from functools import wraps

class redirect_stdout_to_logging:
    """Use as a context manager or decorator to redirect stdout to logging."""

    def __init__(self, level=logging.INFO):
        self.level = level
        self._original_stdout = None

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = LoggerStream(self.level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        return False  # don't suppress exceptions

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


class LoggerStream:
    def __init__(self, level=logging.INFO):
        self.level = level

    def write(self, message):
        if message.strip():
            frame = inspect.currentframe().f_back
            logger_name = frame.f_globals.get('__name__', 'root') if frame else 'root'
            logging.getLogger(logger_name).log(self.level, message.rstrip())

    def flush(self):
        pass
