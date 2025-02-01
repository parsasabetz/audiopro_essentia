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
) -> Tuple[int, Optional[FrameFeatures]]:
    """
    Process a single audio frame to extract various spectral features.

    Args:
        frame_data: Tuple of (frame_index, frame_data)
        sample_rate: The sample rate of the audio signal
        frame_length: The length of the frame to be processed
        window_func: The window function to be applied to the frame
        freq_array: Array of frequency values for FFT bins

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

        # Compute spectrum using Essentia's optimized implementation
        spectrum_alg = es.Spectrum()
        spec = spectrum_alg(frame)

        if np.all(spec == 0):
            return frame_index, None

        # Calculate features using Essentia's optimized algorithms
        centroid = float(es.Centroid(range=sample_rate / 2)(spec))

        # Compute spectral bandwidth with numerical stability
        spectrum_sum = np.sum(spec)  # Keep as float64 for accumulation
        if spectrum_sum <= 1e-10:
            spectral_bandwidth = 0.0
        else:
            # Use float32 for intermediate calculations
            freq_diff = (freq_array - centroid).astype(np.float32)
            spec_float32 = spec.astype(np.float32)
            variance = np.sum(freq_diff * freq_diff * spec_float32) / spectrum_sum
            spectral_bandwidth = float(np.sqrt(np.clip(variance, 0, None)))

        # Extract other features using Essentia
        flatness = float(es.Flatness()(spec))
        rolloff = float(es.RollOff()(spec))
        mfcc_alg = es.MFCC(numberCoefficients=13)
        _, mfcc_coeffs = mfcc_alg(spec)

        # Compute peaks and chroma features
        freqs, mags = es.SpectralPeaks()(spec)
        chroma_vector = (
            es.HPCP()(freqs, mags) if len(freqs) > 0 else np.zeros(12, dtype=np.float32)
        )

        # Compute frequency bands
        freq_bands = compute_frequency_bands(spec, sample_rate, frame_length)

        # Package results in type-safe container
        return frame_index, FrameFeatures(
            time=(frame_index * HOP_LENGTH)
            / sample_rate
            * 1000,  # Time in milliseconds
            rms=float(np.sqrt(np.mean(frame**2))),
            spectral_centroid=centroid,
            spectral_bandwidth=spectral_bandwidth,
            spectral_flatness=flatness,
            spectral_rolloff=rolloff,
            zero_crossing_rate=float(es.ZeroCrossingRate()(frame)),
            mfcc=mfcc_coeffs.tolist(),
            frequency_bands=freq_bands,
            chroma=chroma_vector.tolist(),
        )

    except (ValueError, TypeError, RuntimeError) as e:
        logger.error(f"Frame processing error at {frame_index}: {str(e)}")
        return frame_index, None
    finally:
        # Clean up large arrays
        del spec, frame
