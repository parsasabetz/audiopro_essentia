# Import necessary modules
import signal
import threading
from contextlib import contextmanager

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
