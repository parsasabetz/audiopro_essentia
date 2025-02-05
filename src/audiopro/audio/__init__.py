"""
Audio processing module for feature extraction and analysis.
"""

from .models import FrameFeatures
from .processors import get_frequency_bins, compute_frequency_bands, process_frame
from .extractor import extract_features

__all__ = [
    "FrameFeatures",
    "get_frequency_bins",
    "compute_frequency_bands",
    "process_frame",
    "extract_features",
]
