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
from audiopro.output.types import AVAILABLE_FEATURES, SPECTRAL_FEATURES, FeatureConfig
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
        low_idx = np.searchsorted(freqs, low, side="left")
        high_idx = np.searchsorted(freqs, high, side="right")
        if high_idx > low_idx:
            band_magnitudes = spec[low_idx:high_idx].astype(np.float32)
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
    start_sample: int = 0,  # Add start_sample parameter with default value
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
                      If provided, only features set to True will be computed.
        start_sample: Start sample to offset the frame time (default: 0)

    Returns:
        Tuple of frame index and extracted features, or None if processing failed
    """
    frame_index, frame = frame_data

    try:
        if frame.size == 0 or np.all(np.isnan(frame)):
            return frame_index, None

        # Convert to mono once if multi-channel
        if frame.ndim > 1:
            frame = np.mean(frame, axis=1)

        # Apply window function and pad if necessary
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        frame = frame.astype(np.float32) * window_func

        # Initialize features dictionary and compute RMS once for features
        feature_values = {}
        eps = np.finfo(float).eps
        rms_value = float(np.sqrt(np.mean(frame**2)))

        # Determine enabled features and include "volume" by default if no config is provided
        if feature_config is None:
            enabled_features = AVAILABLE_FEATURES | {"volume"}
        else:
            enabled_features = {k for k, v in feature_config.items() if v}

        if "volume" in enabled_features:
            feature_values["volume"] = 20 * np.log10(rms_value + eps)

        # Compute rms if requested
        if "rms" in enabled_features:
            feature_values["rms"] = rms_value

        # Filter enabled features to only those set to True
        enabled_features = (
            {k for k, v in feature_config.items() if v}
            if feature_config
            else AVAILABLE_FEATURES
        )

        needs_spectrum = bool(enabled_features & SPECTRAL_FEATURES)

        if needs_spectrum:
            spectrum_alg = es.Spectrum()
            spec = spectrum_alg(frame)

            if np.all(spec == 0):
                return frame_index, None

            # Compute centroid once if needed by centroid or bandwidth
            if (
                "spectral_centroid" in enabled_features
                or "spectral_bandwidth" in enabled_features
            ):
                centroid_value = float(es.Centroid(range=sample_rate / 2)(spec))
                if "spectral_centroid" in enabled_features:
                    feature_values["spectral_centroid"] = centroid_value

            if "spectral_bandwidth" in enabled_features:
                spectrum_sum = np.sum(spec)
                if spectrum_sum > 1e-10:
                    # Convert spec once to float32 and reuse for bandwidth computation
                    spec_float32 = spec.astype(np.float32)
                    freq_diff = (freq_array - centroid_value).astype(np.float32)
                    variance = (
                        np.sum(freq_diff * freq_diff * spec_float32) / spectrum_sum
                    )
                    feature_values["spectral_bandwidth"] = float(
                        np.sqrt(np.clip(variance, 0, None))
                    )
                else:
                    feature_values["spectral_bandwidth"] = 0.0

            if "spectral_flatness" in enabled_features:
                flatness_alg = es.Flatness()
                feature_values["spectral_flatness"] = float(flatness_alg(spec))

            if "spectral_rolloff" in enabled_features:
                rolloff_alg = es.RollOff()
                feature_values["spectral_rolloff"] = float(rolloff_alg(spec))

            if "mfcc" in enabled_features:
                mfcc_alg = es.MFCC(numberCoefficients=13)
                _, mfcc_coeffs = mfcc_alg(spec)
                feature_values["mfcc"] = mfcc_coeffs.tolist()

            if "chroma" in enabled_features:
                spectral_peaks = es.SpectralPeaks()
                freqs_peaks, mags_peaks = spectral_peaks(spec)
                hpcp_alg = es.HPCP()
                chroma_vector = (
                    hpcp_alg(freqs_peaks, mags_peaks)
                    if len(freqs_peaks) > 0
                    else np.zeros(12, dtype=np.float32)
                )
                feature_values["chroma"] = chroma_vector.tolist()

            if "frequency_bands" in enabled_features:
                feature_values["frequency_bands"] = compute_frequency_bands(
                    spec, sample_rate, frame_length
                )

        if "zero_crossing_rate" in enabled_features:
            feature_values["zero_crossing_rate"] = float(es.ZeroCrossingRate()(frame))

        # Create FrameFeatures instance with only computed features
        time_ms = ((start_sample + frame_index * HOP_LENGTH) / sample_rate) * 1000
        result = FrameFeatures.create(time=time_ms, **feature_values)
        return frame_index, result

    except (ValueError, TypeError, RuntimeError) as e:
        logger.error(f"Frame processing error at {frame_index}: {str(e)}")
        return frame_index, None
    finally:
        # Rely on garbage collection rather than explicit deletion.
        pass
