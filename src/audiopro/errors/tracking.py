"""Utilities for error tracking and aggregation."""

# typing imports
from typing import Dict, Optional
from dataclasses import dataclass, field

# Third-party imports
import time
from collections import Counter
from collections import deque  # Added for efficient fixed-size queue
from contextlib import contextmanager


@dataclass
class ErrorStats:
    """
    A class to track and record error statistics during audio processing.

    Attributes:
        total_errors (int): The total number of errors recorded.
        error_types (Counter): A counter for different types of errors.
        error_locations (Counter): A counter for locations where errors occurred.
        first_error_time (Optional[float]): The timestamp of the first error occurrence.
        last_error_time (Optional[float]): The timestamp of the last error occurrence.
        error_rate (float): The rate of errors per sample processed.
        samples_processed (int): The number of samples processed.

    Methods:
        record_error(error: Exception, location: str) -> None:
            Records an error occurrence, updating the error statistics.

        get_summary() -> Dict:
            Returns a summary of the error statistics.
    """

    total_errors: int = 0
    error_types: Counter = field(default_factory=Counter)
    error_locations: Counter = field(default_factory=Counter)
    first_error_time: Optional[float] = None
    last_error_time: Optional[float] = None
    error_rate: float = 0.0
    samples_processed: int = 0

    def record_error(self, error: Exception, location: str) -> None:
        """Record an error occurrence."""
        now = time.time()
        if self.first_error_time is None:
            self.first_error_time = now
        self.last_error_time = now

        self.total_errors += 1
        self.error_types[type(error).__name__] += 1
        self.error_locations[location] += 1

        if self.samples_processed > 0:
            self.error_rate = self.total_errors / self.samples_processed

    def get_summary(self) -> Dict:
        """Get error statistics summary."""
        return {
            "total_errors": self.total_errors,
            "error_rate": f"{self.error_rate:.2%}",
            "most_common_error": self.error_types.most_common(1),
            "most_problematic_location": self.error_locations.most_common(1),
            "error_timespan": (
                self.last_error_time - self.first_error_time
                if self.first_error_time
                else 0
            ),
        }


class ErrorRateLimiter:
    """
    A class to limit the rate of error logging.

    Attributes:
        window_size : int
            The time window size in seconds to track errors (default is 1000).
        max_errors : int
            The maximum number of errors allowed within the window size (default is 10).
        error_times : deque
            A deque to store the timestamps of the errors, limited to max_errors.
    """

    def __init__(self, window_size: int = 1000, max_errors: int = 10):
        self.window_size = window_size
        self.max_errors = max_errors
        self.error_times: deque[float] = deque()  # Using deque for efficient pops

    def should_log(self) -> bool:
        """Determine if an error should be logged based on the rate limiting."""
        now = time.time()
        # Pop outdated error timestamps from the left
        while self.error_times and (now - self.error_times[0] > self.window_size):
            self.error_times.popleft()
        if len(self.error_times) < self.max_errors:
            self.error_times.append(now)
            return True
        return False


@contextmanager
def error_tracking_context(stats: ErrorStats):
    """
    Context manager for tracking errors in a processing block.

    Args:
        stats (ErrorStats): An instance of ErrorStats to record errors.

    Yields:
        None

    Raises:
        Exception: Re-raises any exception that occurs within the context block after recording it.
    """
    try:
        yield
    except Exception as e:
        stats.record_error(e, f"{e.__class__.__module__}.{e.__class__.__name__}")
        raise
