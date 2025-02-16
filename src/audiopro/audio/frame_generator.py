"""
Creates a memory-efficient generator for audio frames.
"""

# typing imports
from typing import Iterator, Tuple

# Import necessary modules and types
import numpy as np
from numpy.typing import NDArray

# Import constants for frame length and hop length
from audiopro.utils.constants import (  # pylint: disable=no-name-in-module
    FRAME_LENGTH,
    HOP_LENGTH,
)


def create_frame_generator(
    audio_data: NDArray[np.float32], total_samples: int
) -> Iterator[Tuple[int, NDArray[np.float32]]]:
    """
    Creates a memory-efficient generator for audio frames.

    Args:
        audio_data: The input audio data
        total_samples: Total number of samples in audio

    Yields:
        Tuples of (frame_index, frame_data)
    """
    # Pre-calculate the stop index for range to avoid per-iteration calculation
    stop_idx = total_samples - FRAME_LENGTH + 1

    # Use numpy's strided view for efficient frame extraction
    for frame_idx in range(0, stop_idx, HOP_LENGTH):
        # Use a view instead of a copy for better memory efficiency
        # Only copy if the frame will be sent to another process
        frame_data = audio_data[frame_idx : frame_idx + FRAME_LENGTH]
        if len(frame_data) == FRAME_LENGTH:
            # Only create a copy when yielding since the data will be sent to another process
            yield (frame_idx // HOP_LENGTH, frame_data.copy())
