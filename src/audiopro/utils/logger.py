# typing imports
from typing import Optional

# Standard library imports
import logging
import sys

# Third-party imports
from functools import lru_cache


class LoggerSingleton:
    """Thread-safe singleton logger class with memory optimization."""

    _instance: Optional[logging.Logger] = None
    _log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    @classmethod
    @lru_cache(maxsize=1)  # Cache the logger creation
    def get_logger(cls, name: str = "audiopro") -> logging.Logger:
        """Get or create logger instance with memory optimization."""
        if cls._instance is None:
            cls._instance = logging.getLogger(name)

            if not cls._instance.handlers:
                # Use a single handler and formatter for all loggers
                handler = logging.StreamHandler(sys.stdout)
                formatter = logging.Formatter(cls._log_format)
                handler.setFormatter(formatter)
                cls._instance.addHandler(handler)

                # Set level at logger, not handler (more efficient)
                cls._instance.setLevel(logging.INFO)

                # Prevent propagation for better performance
                cls._instance.propagate = False

        return cls._instance

    @classmethod
    def shutdown(cls):
        """Clean up logging resources."""
        if cls._instance:
            for handler in cls._instance.handlers:
                handler.close()
            cls._instance = None
        logging.shutdown()


# Efficient cached getter function
@lru_cache(maxsize=None)
def get_logger(name: str = "audiopro") -> logging.Logger:
    """Memory-efficient logger accessor."""
    return LoggerSingleton.get_logger(name)


def handle_exception(exc_type, exc_value, exc_traceback):
    """Global exception handler to ensure exceptions are logged"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger = LoggerSingleton.get_logger()
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


# Set the global exception handler
sys.excepthook = handle_exception
