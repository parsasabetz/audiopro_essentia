# Standard library imports
from typing import Dict, Tuple, Any

# Third-party imports
import numpy as np
import essentia.standard as es

# Local application imports
from audiopro.utils.constants import HOP_LENGTH, FREQUENCY_BANDS
from audiopro.utils.logger import get_logger

# Setup logger
logger = get_logger(__name__)

# Cache frequency bins to avoid recomputation
FREQS_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


def compute_frequency_bands(
    spec: np.ndarray, sample_rate: int, frame_length: int
) -> Dict[str, float]:
    """
    Compute the average magnitude of the spectrogram within predefined frequency bands.

    Args:
        spec (np.ndarray): The input spectrogram array.
        sample_rate (int): The sample rate of the audio signal.
        frame_length (int): The length of each frame in the spectrogram.

    Returns:
        Dict[str, float]: A dictionary where the keys are the names of the frequency bands
                          and the values are the average magnitudes within those bands.
    """

    key = (sample_rate, frame_length)
    if key not in FREQS_CACHE:
        # Compute frequency bins for the given sample rate and frame length
        FREQS_CACHE[key] = np.linspace(0, sample_rate / 2, len(spec))
    freqs = FREQS_CACHE[key]

    result = {}
    for band_name, (low, high) in FREQUENCY_BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        result[band_name] = float(np.mean(spec[mask])) if np.any(mask) else 0.0
    return result





def process_frame(
    frame_data: Tuple[int, np.ndarray],
    sample_rate: int,
    frame_length: int,
    window_func: np.ndarray,
    freq_array: np.ndarray,
) -> Tuple[int, Any]:
    """
    Process a single audio frame to extract various spectral features.

    Parameters:
    frame_data (Tuple[int, np.ndarray]): A tuple containing the frame index and the frame data.
    sample_rate (int): The sample rate of the audio signal.
    frame_length (int): The length of the frame to be processed.
    window_func (np.ndarray): The window function to be applied to the frame.
    freq_array (np.ndarray): Array of frequency values corresponding to the FFT bins.

    Returns:
    Tuple[int, Any]: A tuple containing the frame index and a dictionary of extracted features, or None if the frame is invalid.
    The dictionary contains the following keys:
        - "time": The time of the frame in seconds.
        - "rms": The root mean square value of the frame.
        - "spectral_centroid": The spectral centroid of the frame.
        - "spectral_bandwidth": The spectral bandwidth of the frame.
        - "spectral_flatness": The spectral flatness of the frame.
        - "spectral_rolloff": The spectral rolloff of the frame.
        - "zero_crossing_rate": The zero crossing rate of the frame.
        - "mfcc": The Mel-frequency cepstral coefficients (MFCC) of the frame.
        - "frequency_bands": The energy in different frequency bands.
        - "chroma": The chroma vector of the frame.

    Raises:
    ValueError: If there is an issue with the input values.
    TypeError: If there is a type mismatch in the input values.
    RuntimeError: If there is an error during the processing of the frame.
    """

    frame_index, frame = frame_data
    if frame.size == 0 or np.all(np.isnan(frame)):
        return frame_index, None

    try:
        # Apply window function and pad if necessary
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        frame = frame.astype(np.float32) * window_func

        # Compute spectrum
        spectrum_alg = es.Spectrum()
        spec = spectrum_alg(frame)
        if np.all(spec == 0):
            return frame_index, None

        # Calculate spectral centroid using Essentia (using half-range as defined)
        centroid = es.Centroid(range=sample_rate / 2)(spec)
        spectrum_sum = np.sum(spec)
        # Compute spectral bandwidth manually with numerical stability
        if spectrum_sum <= 1e-10:
            spectral_bandwidth = 0.0
        else:
            freq_diff = freq_array - centroid
            variance = np.sum(freq_diff * freq_diff * spec) / spectrum_sum
            spectral_bandwidth = float(np.sqrt(np.clip(variance, 0, None)))

        # Other features
        flatness = es.Flatness()(spec)
        rolloff = es.RollOff()(spec)
        mfcc_alg = es.MFCC(numberCoefficients=13)
        _, mfcc_out = mfcc_alg(spec)
        freqs, mags = es.SpectralPeaks()(spec)
        chroma_vector = es.HPCP()(freqs, mags) if len(freqs) > 0 else np.zeros(12)
        freq_bands = compute_frequency_bands(spec, sample_rate, frame_length)

        return frame_index, {
            "time": frame_index * HOP_LENGTH / sample_rate,
            "rms": float(np.sqrt(np.mean(frame**2))),
            "spectral_centroid": centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "spectral_flatness": flatness,
            "spectral_rolloff": rolloff,
            "zero_crossing_rate": es.ZeroCrossingRate()(frame),
            "mfcc": mfcc_out.tolist(),
            "frequency_bands": freq_bands,
            "chroma": chroma_vector.tolist(),
        }
    except (ValueError, TypeError, RuntimeError) as e:
        logger.error("Frame processing error at %d: %s", frame_index, str(e))
        return frame_index, None
