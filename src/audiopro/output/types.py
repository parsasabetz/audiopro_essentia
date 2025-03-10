"""Type definitions for audio analysis output."""

# typing imports
from typing import List, TypedDict, Optional, Literal

# Consolidate feature definitions into a single source of truth
FEATURE_DEFINITIONS = {
    "rms": "Root Mean Square energy value",
    "volume": "Volume in decibels (20 * log10(rms))",
    "spectral_centroid": "Weighted mean of frequencies",
    "spectral_bandwidth": "Variance of frequencies around the centroid",
    "spectral_flatness": "Measure of how noise-like the signal is",
    "spectral_rolloff": "Frequency below which most spectral energy exists",
    "zero_crossing_rate": "Rate of signal polarity changes",
    "frequency_bands": "Energy in different frequency bands",
    "mfcc": "Mel-frequency cepstral coefficients (13 values)",
    "chroma": "Distribution of spectral energy across pitch classes",
}

FEATURE_NAMES = tuple(FEATURE_DEFINITIONS.keys())

AVAILABLE_FEATURES = frozenset(FEATURE_NAMES)

SPECTRAL_FEATURES = frozenset(
    f for f in AVAILABLE_FEATURES if f not in {"rms", "zero_crossing_rate", "volume"}
)

# Create Literal type from feature names
FeatureName = Literal[
    "rms",
    "volume",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_flatness",
    "spectral_rolloff",
    "zero_crossing_rate",
    "frequency_bands",
    "mfcc",
    "chroma",
]

# Add MimeType Literal for autocomplete support
MimeType = Literal[
    "audio/wav",
    "audio/mpeg",
    "audio/ogg",
    "audio/flac",
    "audio/mp4",
    "audio/aiff",
    "unknown",
]


# Define a TypedDict for time range (start_time, end_time)
class TimeRange(TypedDict, total=False):
    """Time range specification for audio analysis.

    Attributes:
        start: Start time in seconds
        end: Optional end time in seconds. If not provided, processes until the end.
    """

    start: float
    end: Optional[float]


class FeatureConfig(TypedDict, total=False):
    """Configuration for which audio features to extract."""

    __annotations__ = {feature: bool for feature in FEATURE_NAMES}


def create_feature_config(
    selected_features: Optional[List[str]] = None,
) -> Optional[FeatureConfig]:
    """
    Create a feature configuration dictionary based on selected features.
    Note: selected_features are assumed to be pre-validated by argparse.

    Args:
        selected_features: List of feature names to enable. If None, all features will be computed.
                         Must be a subset of `AVAILABLE_FEATURES`.

    Returns:
        Optional[FeatureConfig]: Configuration with selected features set to `True`, others to `False`.
                               None if no features were selected (compute all).
    """
    if selected_features is None:
        # When no features are specified, enable all features
        return {feature: True for feature in AVAILABLE_FEATURES}

    # Create config with all features explicitly set to False by default
    config = {feature: False for feature in AVAILABLE_FEATURES}

    # Enable only the selected features
    for feature in selected_features:
        config[feature] = True

    return config


class QualityMetrics(TypedDict):
    dc_offset: float
    silence_ratio: float
    potentially_clipped_samples: int


class FileInfo(TypedDict):
    filename: str
    format: str
    codec: str
    size_mb: float
    created_date: str  # Unix timestamp
    mime_type: MimeType  # MIME type with autocomplete support
    md5_hash: str


class AudioInfo(TypedDict):
    duration_seconds: float
    sample_rate: int
    bit_rate: int
    channels: int
    peak_amplitude: float
    rms_amplitude: float
    dynamic_range_db: float
    quality_metrics: QualityMetrics


class Metadata(TypedDict):
    file_info: FileInfo
    audio_info: AudioInfo


class FrequencyBands(TypedDict, total=False):
    sub_bass: float  # 20-60 Hz
    bass: float  # 60-250 Hz
    low_mid: float  # 250-500 Hz
    mid: float  # 500-2000 Hz
    upper_mid: float  # 2000-5000 Hz
    treble: float  # 5000-20000 Hz


class AudioFeature(TypedDict, total=False):
    """A TypedDict containing audio features extracted from an audio signal.

    Each instance represents a single frame of audio analysis with various acoustic measurements.
    Features will only be present if they were requested in the FeatureConfig.

    Attributes:
        time (float): Unix timestamp in seconds for the current frame.
        rms (Optional[float]): Root Mean Square energy value.
        volume (Optional[float]): Volume in decibels (20 * log10(rms)).
        spectral_centroid (Optional[float]): Weighted mean of frequencies present in the signal.
        spectral_bandwidth (Optional[float]): Variance of frequencies around the spectral centroid.
        spectral_flatness (Optional[float]): Measure of how noise-like the signal is (0=pure tone, 1=noise).
        spectral_rolloff (Optional[float]): Frequency below which a certain percentage of spectral energy exists.
        zero_crossing_rate (Optional[float]): Rate at which signal changes from positive to negative or vice versa.
        mfcc (Optional[List[float]]): Mel-frequency cepstral coefficients (13 values).
        frequency_bands (Optional[FrequencyBands]): Energy distribution across frequency bands.
        chroma (Optional[List[float]]): Distribution of spectral energy across the 12 pitch classes (12 values).
    """

    time: float  # Unix timestamp in seconds
    rms: Optional[float]
    volume: Optional[float]
    spectral_centroid: Optional[float]
    spectral_bandwidth: Optional[float]
    spectral_flatness: Optional[float]
    spectral_rolloff: Optional[float]
    zero_crossing_rate: Optional[float]
    frequency_bands: Optional[FrequencyBands]
    mfcc: Optional[List[float]]  # 13 coefficients
    chroma: Optional[List[float]]  # 12 values


class AudioAnalysis(TypedDict):
    """Audio analysis results containing metadata, tempo, beats, and extracted features.

    Attributes:
        metadata (Metadata): File and audio information
        tempo (float): Detected tempo in BPM
        beats (List[float]): Beat positions in seconds
        features (List[AudioFeature]): List of frame-by-frame feature measurements
        included_features (List[FeatureName]): List of feature names included in the analysis.
                                     Empty list means all available features were computed.
    """

    metadata: Metadata
    tempo: float
    included_features: List[FeatureName]
    beats: List[float]
    features: List[AudioFeature]


class LoaderMetadata(TypedDict):
    """Type specification for audio file metadata.

    A TypedDict class that defines metadata attributes for loaded audio files.

    Attributes:
        filename (str): Name of the audio file.
        format (str): Audio file format (e.g., 'wav', 'mp3').
        size_mb (float): File size in megabytes.
        created_date (float): File creation timestamp.
        mime_type (MimeType): MIME type of the audio file.
        md5_hash (str): MD5 hash of file contents.
        bit_rate (int): Audio bit rate in bits per second.
        codec (str): Audio codec used for encoding.
        channels (int): Number of audio channels.
        sample_rate (int): Sample rate in Hz.
    """

    filename: str
    format: str
    size_mb: float
    created_date: float  # timestamp
    mime_type: MimeType
    md5_hash: str
    bit_rate: int
    codec: str
    channels: int
    sample_rate: int
