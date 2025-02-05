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

from .tracking import (
    ErrorRateLimiter,
    ErrorStats,
    error_tracking_context,
)

__all__ = [
    # exceptions
    "AudioIOError",
    "AudioProcessingError",
    "AudioValidationError",
    "FeatureExtractionError",
    "SpectralFeatureError",
    # tracking
    "ErrorRateLimiter",
    "ErrorStats",
    "error_tracking_context",
]
