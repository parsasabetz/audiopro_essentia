"""Output handling module."""

from .output_handler import write_output
from .types import (
    AudioAnalysis,
    AudioFeature,
    Metadata,
    FileInfo,
    AudioInfo,
    FrequencyBands,
    QualityMetrics,
    LoaderMetadata,
    TimeRange,
    FeatureConfig,
    FEATURE_NAMES,
    AVAILABLE_FEATURES,
    SPECTRAL_FEATURES,
)
from .feature_flags import FeatureFlagSet, create_feature_flags
from .modules import _write_json, _write_msgpack


__all__ = [
    "write_output",
    "AudioAnalysis",
    "AudioFeature",
    "Metadata",
    "FileInfo",
    "AudioInfo",
    "FrequencyBands",
    "QualityMetrics",
    "_write_json",
    "_write_msgpack",
    "LoaderMetadata",
    "SPECTRAL_FEATURES",
    "TimeRange",
    "FeatureConfig",
    "FeatureFlagSet",
    "create_feature_flags",
    "FEATURE_NAMES",
    "AVAILABLE_FEATURES",
]
