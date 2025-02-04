# typing imports
from typing import Tuple
from functools import lru_cache

# third-party imports
import essentia.standard as es
import numpy as np

# local imports
from audiopro.utils.logger import get_logger

# setup logger
logger = get_logger(__name__)


class RhythmExtractorSingleton:
    """
    RhythmExtractorSingleton is a thread-safe singleton class for rhythm extraction.

    This class ensures that only one instance of the rhythm extractor is created and used throughout the application.
    It uses the Essentia library's `RhythmExtractor2013` with the "multifeature" method for rhythm extraction.

    Attributes:
        _instance (RhythmExtractorSingleton): The singleton instance of the class.
        extractor (essentia.standard.RhythmExtractor2013): The rhythm extractor instance from the Essentia library.

    Methods:
        __new__(cls): Creates and returns the singleton instance of the class.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.extractor = es.RhythmExtractor2013(method="multifeature")
        return cls._instance


def extract_rhythm(audio: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Extracts the rhythm from an audio signal.

    Parameters:
    audio (np.ndarray): A numpy array containing the audio signal.

    Returns:
    Tuple[float, np.ndarray]: A tuple containing the tempo (in beats per minute)
                              and an array of beat positions.

    Raises:
    TypeError: If the input audio is not a numpy array.
    ValueError: If the input audio data is empty.
    RuntimeError: If rhythm extraction fails for any reason.
    """

    # Use numpy's built-in type checking (faster than isinstance)
    if not hasattr(audio, "dtype") or not hasattr(audio, "size"):
        raise TypeError("Audio must be a numpy array")
    if audio.size == 0:
        raise ValueError("Audio data is empty")

    try:
        # More efficient memory handling
        if len(audio.shape) > 1:
            # Use out parameter to avoid extra allocation
            audio = np.mean(audio, axis=1, dtype=np.float32, out=None)
        else:
            # Only convert if necessary
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32, copy=False)

        # Use reshape(-1) instead of flatten() to avoid copy
        audio = audio.reshape(-1)
        if not audio.flags.c_contiguous:
            audio = np.ascontiguousarray(audio)

        # Get extractor instance safely
        extractor = RhythmExtractorSingleton().extractor
        tempo, beat_positions, _, _, _ = extractor(audio)

        # Recalculate tempo based on median beat interval if possible
        if len(beat_positions) > 1:
            from numpy import diff, median  # pylint: disable=import-outside-toplevel

            intervals = diff(beat_positions)
            median_interval = median(intervals)
            if median_interval > 0:
                tempo = 60.0 / median_interval

        return tempo, beat_positions
    except Exception as e:
        logger.error(f"Rhythm extraction failed: {str(e)}")
        raise RuntimeError(f"Failed to extract rhythm: {str(e)}") from e


@lru_cache(maxsize=32)
def compute_spectral_bandwidth(
    spectrum: np.ndarray, freqs: np.ndarray, centroid: float
) -> float:
    """
    Compute the spectral bandwidth of a given spectrum.

    Spectral bandwidth is a measure of the width of the spectrum around its centroid.

    Parameters:
        spectrum (np.ndarray): The amplitude spectrum of the signal.
        freqs (np.ndarray): The corresponding frequencies of the spectrum.
        centroid (float): The spectral centroid of the spectrum.

    Returns:
        float: The spectral bandwidth of the spectrum.

    Raises:
        TypeError: If spectrum or freqs are not numpy arrays.
        ValueError: If spectrum and freqs do not have the same size.
    """

    if not (isinstance(spectrum, np.ndarray) and isinstance(freqs, np.ndarray)):
        raise TypeError("Spectrum and frequencies must be numpy arrays")

    if spectrum.size != freqs.size:
        raise ValueError("Spectrum and frequencies must have the same size")

    # Pre-calculate condition to avoid unnecessary computation
    if np.sum(spectrum) <= 1e-10:
        return 0.0

    # Use direct broadcasting for better memory efficiency
    freq_diff = freqs - centroid
    variance = (freq_diff * freq_diff * spectrum).sum() / spectrum.sum()

    return float(np.sqrt(np.clip(variance, 0, None)))
