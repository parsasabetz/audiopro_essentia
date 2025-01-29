import essentia.standard as es
import warnings
import numpy as np


def extract_rhythm(audio: any) -> tuple:
    """Extract rhythm features from audio data."""
    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    tempo, beat_positions, _, _, _ = rhythm_extractor(audio)
    return tempo, beat_positions


def compute_spectral_bandwidth(
    spectrum: np.ndarray, freqs: np.ndarray, centroid: float
) -> float:
    """
    Manually compute the spectral bandwidth.

    Args:
        spectrum: Magnitude spectrum of the audio frame.
        freqs: Frequency bins corresponding to the spectrum.
        centroid: Spectral centroid of the audio frame.

    Returns:
        Spectral bandwidth as a float.
    """
    # Calculate the variance around the centroid
    variance = np.sum(((freqs - centroid) ** 2) * spectrum) / np.sum(spectrum)
    # Return the standard deviation as spectral bandwidth
    return np.sqrt(variance)
