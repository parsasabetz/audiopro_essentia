"""Type definitions for audio analysis output."""

from typing import List, TypedDict


class QualityMetrics(TypedDict):
    dc_offset: float
    silence_ratio: float
    potentially_clipped_samples: int


class FileInfo(TypedDict):
    filename: str
    format: str
    size_mb: float
    created_date: str  # ISO format
    mime_type: str
    sha256_hash: str


class AudioInfo(TypedDict):
    duration_seconds: float
    sample_rate: int
    channels: int
    peak_amplitude: float
    rms_amplitude: float
    dynamic_range_db: float
    quality_metrics: QualityMetrics


class Metadata(TypedDict):
    file_info: FileInfo
    audio_info: AudioInfo


class FrequencyBands(TypedDict):
    sub_bass: float  # 20-60 Hz
    bass: float  # 60-250 Hz
    low_mid: float  # 250-500 Hz
    mid: float  # 500-2000 Hz
    upper_mid: float  # 2000-5000 Hz
    treble: float  # 5000-20000 Hz


class AudioFeature(TypedDict):
    """A TypedDict containing audio features extracted from an audio signal.

    Each instance represents a single frame of audio analysis with various acoustic measurements.

    Attributes:
        time (float): Time position in seconds for the current frame.
        rms (float): Root Mean Square energy value.
        spectral_centroid (float): Weighted mean of frequencies present in the signal.
        spectral_bandwidth (float): Variance of frequencies around the spectral centroid.
        spectral_flatness (float): Measure of how noise-like the signal is (0=pure tone, 1=noise).
        spectral_rolloff (float): Frequency below which a certain percentage of spectral energy exists.
        zero_crossing_rate (float): Rate at which signal changes from positive to negative or vice versa.
        mfcc (List[float]): Mel-frequency cepstral coefficients (13 values).
        frequency_bands (FrequencyBands): Energy distribution across frequency bands.
        chroma (List[float]): Distribution of spectral energy across the 12 pitch classes (12 values).
    """
    time: float
    rms: float
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_flatness: float
    spectral_rolloff: float
    zero_crossing_rate: float
    mfcc: List[float]  # 13 coefficients
    frequency_bands: FrequencyBands
    chroma: List[float]  # 12 values


class AudioAnalysis(TypedDict):
    metadata: Metadata
    tempo: float
    beats: List[float]
    features: List[AudioFeature]
