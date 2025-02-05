"""
Audio processing errors module.
"""

from .exceptions import (
    AudioIOError,
    AudioProcessingError,
    AudioValidationError,
    FeatureExtractionError,
    SpectralFeatureError,
)

__all__ = [
    "AudioIOError",
    "AudioProcessingError",
    "AudioValidationError",
    "FeatureExtractionError",
    "SpectralFeatureError",
]
