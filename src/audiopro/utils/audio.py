# typing imports
from typing import Tuple

# third-party imports
import essentia.standard as es
import numpy as np


class RhythmExtractorSingleton:
    """
    RhythmExtractorSingleton is a thread-safe singleton class for rhythm extraction.

    This class ensures that only one instance of the rhythm extractor is created and used throughout the application. 
    It uses the Essentia library's RhythmExtractor2013 with the "multifeature" method for rhythm extraction.

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
        # Get extractor instance safely
        extractor = RhythmExtractorSingleton().extractor
        tempo, beat_positions, _, _, _ = extractor(audio)
        return tempo, beat_positions
    except Exception as e:
        raise RuntimeError(f"Failed to extract rhythm: {str(e)}") from e


def compute_spectral_bandwidth(
    spectrum: np.ndarray, freqs: np.ndarray, centroid: float
) -> float:
    """
    Manually compute the spectral bandwidth.
    Parameters:
        spectrum (np.ndarray): The amplitude spectrum of the signal.
        freqs (np.ndarray): The corresponding frequencies of the spectrum.
        centroid (float): The spectral centroid of the signal.

    Returns:
        float: The computed spectral bandwidth.

    Raises:
        TypeError: If spectrum or freqs are not numpy arrays.
        ValueError: If spectrum and freqs do not have the same size.
    """
    # Fast array check
    if not (hasattr(spectrum, "dtype") and hasattr(freqs, "dtype")):
        raise TypeError("Spectrum and frequencies must be numpy arrays")

    # Use view instead of copy for efficiency
    if spectrum.size != freqs.size:
        raise ValueError("Spectrum and frequencies must have the same size")

    # Cache sum to avoid recalculation
    spectrum_sum = np.sum(spectrum)
    if spectrum_sum <= 1e-10:  # More numerically stable threshold
        return 0.0

    # Vectorized computation without intermediate arrays
    freq_diff = freqs - centroid
    variance = np.sum(freq_diff * freq_diff * spectrum) / spectrum_sum

    # Use np.sqrt with clip for better numerical stability
    return float(np.sqrt(np.clip(variance, 0, None)))
