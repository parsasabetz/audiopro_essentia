# typing imports
from typing import Dict
from functools import lru_cache

# Third-party imports
import numpy as np
from numpy.typing import NDArray
import torch

# Local application imports
from audiopro.utils.logger import get_logger
from audiopro.utils.constants import (  # pylint: disable=no-name-in-module
    FRAME_LENGTH,
    FREQUENCY_BANDS,
)

# Setup logger
logger = get_logger(__name__)

# Pre-compute constants for efficiency
EPS: float = torch.finfo(torch.float32).eps


@lru_cache(maxsize=32)
def get_frequency_bins(sample_rate: int) -> NDArray[np.float32]:
    """
    Cached computation of frequency bins.

    Args:
        sample_rate: The sample rate of the audio

    Returns:
        Array of frequency bins
    """
    return np.linspace(0, sample_rate / 2, FRAME_LENGTH // 2 + 1, dtype=np.float32)


def compute_frequency_bands(
    spec: NDArray[np.float32], sample_rate: int
) -> Dict[str, float]:
    """
    Compute the average magnitude of the spectrogram within predefined frequency bands.

    Args:
        spec: The input spectrogram array (power spectrum)
        sample_rate: The sample rate of the audio signal

    Returns:
        Dictionary mapping frequency band names to their average magnitudes
    """
    logger.debug(f"Input spec shape: {spec.shape}")

    # Ensure spec is 1D and handle empty or invalid input
    if spec.size == 0:
        logger.error("Empty spectrogram received")
        return {band: 0.0 for band in FREQUENCY_BANDS}

    # Get frequency bins
    freqs = get_frequency_bins(sample_rate)
    logger.debug(f"Frequency bins shape: {freqs.shape}")

    # Ensure spec has the correct shape
    if spec.shape[0] != freqs.shape[0]:
        logger.error(f"Shape mismatch: spec {spec.shape}, freqs {freqs.shape}")
        if spec.shape[0] < freqs.shape[0]:
            spec = np.pad(spec, (0, freqs.shape[0] - spec.shape[0]))
        else:
            spec = spec[: freqs.shape[0]]
        logger.debug(f"Adjusted spec shape: {spec.shape}")

    result = {}

    # Apply log scaling to better handle the dynamic range
    spec_db = 10 * np.log10(np.maximum(spec, EPS))

    # Normalize to dB scale
    spec_db = np.clip(spec_db, -80, 0)  # Clip to -80 dB
    spec_db = spec_db + 80  # Shift to positive range
    spec_db = spec_db / 80  # Normalize to [0, 1]

    for band_name, (low, high) in FREQUENCY_BANDS.items():
        low_idx = np.searchsorted(freqs, low, side="left")
        high_idx = np.searchsorted(freqs, high, side="right")

        # Ensure valid indices
        low_idx = min(low_idx, len(freqs) - 1)
        high_idx = min(high_idx, len(freqs))

        if high_idx > low_idx:
            try:
                # Calculate band energy
                band_magnitudes = spec_db[low_idx:high_idx]
                band_freqs = freqs[low_idx:high_idx]

                # Use frequency-weighted average
                weights = np.log10(band_freqs / (low + EPS) + 1)
                weights = weights / (np.sum(weights) + EPS)  # Normalize weights

                band_energy = np.sum(band_magnitudes * weights)
                result[band_name] = float(band_energy)
            except (ValueError, IndexError, TypeError) as e:
                logger.error(f"Error processing band {band_name}: {str(e)}")
                result[band_name] = 0.0
        else:
            result[band_name] = 0.0

    # Normalize the results to sum to 1
    total_energy = sum(result.values()) + EPS
    result = {k: v / total_energy for k, v in result.items()}

    return result
