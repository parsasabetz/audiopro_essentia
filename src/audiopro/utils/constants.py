"""
Constants for audio processing.
"""

# typing imports
from typing import ClassVar, Dict, Tuple
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AudioConstants:
    """
    A class to hold constants used in audio processing.

    Attributes:
        FRAME_LENGTH (int): The length of each audio frame.
        HOP_LENGTH (int): The number of samples between successive frames.
        BATCH_SIZE (int): The number of audio frames to process in a batch.
        FREQUENCY_BANDS (Dict[str, Tuple[int, int]]): A dictionary mapping frequency band names to their respective frequency ranges in Hz.
        FREQUENCY_RANGES (Tuple[Tuple[int, int], ...]): A tuple of frequency ranges derived from FREQUENCY_BANDS.
    """

    FRAME_LENGTH: ClassVar[int] = 2_048
    HOP_LENGTH: ClassVar[int] = 512
    BATCH_SIZE: ClassVar[int] = 1_000
    FREQUENCY_BANDS: ClassVar[Dict[str, Tuple[int, int]]] = {
        "sub_bass": (20, 60),
        "bass": (60, 250),
        "low_mid": (250, 500),
        "mid": (500, 2000),
        "upper_mid": (2000, 5000),
        "treble": (5000, 20000),
    }
    FREQUENCY_RANGES: ClassVar[Tuple[Tuple[int, int], ...]] = tuple(
        FREQUENCY_BANDS.values()
    )


# Single global instance with minimal memory footprint
AUDIO_CONSTANTS = AudioConstants()


# Use properties instead of re-exports to avoid duplicate references
def __getattr__(name):
    if hasattr(AUDIO_CONSTANTS, name):
        return getattr(AUDIO_CONSTANTS, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
