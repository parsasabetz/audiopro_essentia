# typing imports
from typing import Dict, Optional, Tuple

# Standard library imports
from functools import lru_cache

# Third-party imports
import numpy as np
from numpy.typing import NDArray
import essentia.standard as es

# Local application imports
from audiopro.utils.constants import HOP_LENGTH, FREQUENCY_BANDS
from audiopro.utils.logger import get_logger
from audiopro.output.types import FeatureConfig
from .models import FrameFeatures

# Setup logger
logger = get_logger(__name__)


@lru_cache(maxsize=32)
def get_frequency_bins(sample_rate: int, frame_length: int) -> NDArray[np.float32]:
    """
    Cached computation of frequency bins.

    Args:
        sample_rate: The sample rate of the audio
        frame_length: The length of each frame

    Returns:
        Array of frequency bins
    """
    return np.linspace(0, sample_rate / 2, frame_length // 2 + 1, dtype=np.float32)


def compute_frequency_bands(
    spec: NDArray[np.float32], sample_rate: int, frame_length: int
) -> Dict[str, float]:
    """
    Compute the average magnitude of the spectrogram within predefined frequency bands.

    Args:
        spec: The input spectrogram array
        sample_rate: The sample rate of the audio signal
        frame_length: The length of each frame in the spectrogram

    Returns:
        Dictionary mapping frequency band names to their average magnitudes
    """
    freqs = get_frequency_bins(sample_rate, frame_length)

    result: Dict[str, float] = {}
    for band_name, (low, high) in FREQUENCY_BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            # Use float32 for intermediate calculations
            band_magnitudes = spec[mask].astype(np.float32)
            result[band_name] = float(np.mean(band_magnitudes))
        else:
            result[band_name] = 0.0
    return result


def process_frame(
    frame_data: Tuple[int, NDArray[np.float32]],
    sample_rate: int,
    frame_length: int,
    window_func: NDArray[np.float32],
    freq_array: NDArray[np.float32],
    feature_config: Optional[FeatureConfig] = None,
) -> Tuple[int, Optional[FrameFeatures]]:
    """
    Process a single audio frame to extract various spectral features.

    Args:
        frame_data: Tuple of (frame_index, frame_data)
        sample_rate: The sample rate of the audio signal
        frame_length: The length of the frame to be processed
        window_func: The window function to be applied to the frame
        freq_array: Array of frequency values for FFT bins
        feature_config: Optional configuration specifying which features to compute.
                      If None, all features will be computed.

    Returns:
        Tuple of frame index and extracted features, or None if processing failed
    """
    frame_index, frame = frame_data

    try:
        if frame.size == 0 or np.all(np.isnan(frame)):
            return frame_index, None

        # Apply window function and pad if necessary
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        frame = frame.astype(np.float32) * window_func

        # Initialize a dictionary to store computed features
        feature_values = {"time": (frame_index * HOP_LENGTH) / sample_rate * 1000}

        # Compute spectrum only if needed for any spectral features
        needs_spectrum = (
            any(
                feature_config.get(feature, True)  # Default to True if config is None
                for feature in [
                    "spectral_centroid",
                    "spectral_bandwidth",
                    "spectral_flatness",
                    "spectral_rolloff",
                    "mfcc",
                    "frequency_bands",
                    "chroma",
                ]
            )
            if feature_config is not None
            else True
        )

        if needs_spectrum:
            # Compute spectrum using Essentia's optimized implementation
            spectrum_alg = es.Spectrum()
            spec = spectrum_alg(frame)

            if np.all(spec == 0):
                return frame_index, None

            # Calculate features using Essentia's optimized algorithms
            if feature_config is None or feature_config.get("spectral_centroid"):
                feature_values["spectral_centroid"] = float(
                    es.Centroid(range=sample_rate / 2)(spec)
                )

            if feature_config is None or feature_config.get("spectral_bandwidth"):
                # Compute spectral bandwidth with numerical stability
                spectrum_sum = np.sum(spec)  # Keep as float64 for accumulation
                if spectrum_sum <= 1e-10:
                    feature_values["spectral_bandwidth"] = 0.0
                else:
                    # Use float32 for intermediate calculations
                    freq_diff = (
                        freq_array - (feature_values.get("spectral_centroid", 0.0))
                    ).astype(np.float32)
                    spec_float32 = spec.astype(np.float32)
                    variance = (
                        np.sum(freq_diff * freq_diff * spec_float32) / spectrum_sum
                    )
                    feature_values["spectral_bandwidth"] = float(
                        np.sqrt(np.clip(variance, 0, None))
                    )

            if feature_config is None or feature_config.get("spectral_flatness"):
                feature_values["spectral_flatness"] = float(es.Flatness()(spec))

            if feature_config is None or feature_config.get("spectral_rolloff"):
                feature_values["spectral_rolloff"] = float(es.RollOff()(spec))

            if feature_config is None or feature_config.get("mfcc"):
                mfcc_alg = es.MFCC(numberCoefficients=13)
                _, mfcc_coeffs = mfcc_alg(spec)
                feature_values["mfcc"] = mfcc_coeffs.tolist()

            if feature_config is None or feature_config.get("chroma"):
                freqs, mags = es.SpectralPeaks()(spec)
                chroma_vector = (
                    es.HPCP()(freqs, mags)
                    if len(freqs) > 0
                    else np.zeros(12, dtype=np.float32)
                ).tolist()
                feature_values["chroma"] = chroma_vector

            if feature_config is None or feature_config.get("frequency_bands"):
                feature_values["frequency_bands"] = compute_frequency_bands(
                    spec, sample_rate, frame_length
                )

        if feature_config is None or feature_config.get("rms"):
            feature_values["rms"] = float(np.sqrt(np.mean(frame**2)))

        if feature_config is None or feature_config.get("zero_crossing_rate"):
            feature_values["zero_crossing_rate"] = float(es.ZeroCrossingRate()(frame))

        # Create FrameFeatures instance with only computed features
        return frame_index, FrameFeatures(**feature_values)

    except (ValueError, TypeError, RuntimeError) as e:
        logger.error(f"Frame processing error at {frame_index}: {str(e)}")
        return frame_index, None
    finally:
        # Clean up large arrays
        if needs_spectrum:
            del spec
        del frame
