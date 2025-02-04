# Import necessary modules
import signal
import threading
from contextlib import contextmanager
from functools import lru_cache

# Import custom logger
from audiopro.utils import get_logger

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
def calculate_max_workers(
    audio_data_length: int, frame_length: int, hop_length: int
) -> int:
    """
    Calculate the maximum number of workers for processing audio data.

    This function determines the number of workers based on the length of the audio data,
    the frame length, and the hop length. The number of workers is constrained to be
    between 1 and 32, inclusive.

    Args:
        audio_data_length (int): The total length of the audio data.
        frame_length (int): The length of each frame.
        hop_length (int): The hop length between frames.

    Returns:
        int: The calculated number of workers, constrained to a minimum of 1 and a maximum of 32.
    """

    num_frames = (audio_data_length - frame_length) // hop_length + 1
    return min(32, max(1, num_frames // 1000))
