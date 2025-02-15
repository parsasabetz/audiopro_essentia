"""
This module contains utility functions for handling CPU-bound processing of audio data.
"""

# Import CPU-bound functions and utilities for processing audio data
import signal
import threading

# Import context manager for graceful shutdown
from contextlib import contextmanager
from functools import lru_cache

# Import local utilities
from audiopro.utils import get_logger
from audiopro.utils.constants import (  # pylint: disable=no-name-in-module
    FRAME_LENGTH,
    HOP_LENGTH,
)

# Set up logging
logger = get_logger(__name__)


@contextmanager
def graceful_shutdown():
    """
    Context manager to handle graceful shutdown of a process.

    This function sets up signal handlers for SIGTERM and SIGINT to allow for
    a clean shutdown of the process. When a shutdown signal is received, it
    sets a stop flag that can be used to terminate ongoing operations gracefully.

    Yields:
        threading.Event: An event that is set when a shutdown signal is received.

    Example:
        with graceful_shutdown() as stop_flag:
            while not stop_flag.is_set():
                # Perform ongoing operations
                pass
    """

    original_handlers = {}
    stop_flag = threading.Event()

    def handler(_signum, _frame):
        logger.info("Received shutdown signal, cleaning up...")
        stop_flag.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        original_handlers[sig] = signal.signal(sig, handler)

    try:
        yield stop_flag
    finally:
        for sig, original_handler in original_handlers.items():
            signal.signal(sig, original_handler)


@lru_cache(maxsize=128)
def calculate_max_workers(audio_data_length: int) -> int:
    """
    Calculate the maximum number of workers for processing audio data.

    This function determines the number of workers based on the length of the audio data
    and the constants defined for frame and hop length. The number of workers is constrained
    to be between 1 and 32, inclusive.

    Args:
        audio_data_length (int): The total length of the audio data.

    Returns:
        int: The calculated number of workers, constrained to a minimum of 1 and a maximum of 32.
    """
    num_frames = (audio_data_length - FRAME_LENGTH) // HOP_LENGTH + 1
    return min(32, max(1, num_frames // 1000))
