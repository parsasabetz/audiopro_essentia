# typing imports
from typing import Dict

# Standard library imports
import logging
import sys

# Third-party imports
from functools import lru_cache


class LoggerSingleton:
    """
    LoggerSingleton is a thread-safe singleton logger class with memory optimization.

    Class Attributes:
        _loggers (Dict[str, logging.Logger]): A dictionary to store logger instances.
        _formatter (logging.Formatter): A formatter for log messages.
        _handler (logging.Handler): A handler for log messages.

    Class Methods:
        _get_handler() -> logging.Handler:
            Returns a stream handler for logging, creating it if it doesn't exist.
        
        get_logger(name: str = "audiopro") -> logging.Logger:
            Returns a logger instance with the specified name, creating it if it doesn't exist.
            The logger is configured with a stream handler and a specific formatter.
        
        shutdown():
            Cleans up logging resources by closing the handler and clearing the loggers dictionary.
    """

    _loggers: Dict[str, logging.Logger] = {}
    _formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    _handler = None

    @classmethod
    def _get_handler(cls):
        if cls._handler is None:
            cls._handler = logging.StreamHandler(sys.stdout)
            cls._handler.setFormatter(cls._formatter)
        return cls._handler

    @classmethod
    @lru_cache(maxsize=None)
    def get_logger(cls, name: str = "audiopro") -> logging.Logger:
        """Get or create logger instance with proper module hierarchy."""
        if name not in cls._loggers:
            logger = logging.getLogger(name)

            if not logger.handlers:
                logger.addHandler(cls._get_handler())
                logger.setLevel(logging.INFO)
                logger.propagate = False

            cls._loggers[name] = logger

        return cls._loggers[name]

    @classmethod
    def shutdown(cls):
        """Clean up logging resources."""
        if cls._handler:
            cls._handler.close()
            cls._handler = None
        cls._loggers.clear()
        logging.shutdown()


# Efficient cached getter function
@lru_cache(maxsize=None)
def get_logger(name: str = "audiopro") -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name (str): The name of the logger. Defaults to "audiopro".

    Returns:
        logging.Logger: A logger instance with the specified name.
    """
    return LoggerSingleton.get_logger(name)


def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Global exception handler to ensure exceptions are logged.

    This function will be used as a global exception handler to log any uncaught exceptions.
    If the exception is a KeyboardInterrupt, it will call the default exception hook.
    Otherwise, it will log the exception details using the LoggerSingleton.

    Args:
        exc_type (type): The exception type.
        exc_value (Exception): The exception instance.
        exc_traceback (traceback): The traceback object.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger = LoggerSingleton.get_logger()
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


# Set the global exception handler
sys.excepthook = handle_exception
