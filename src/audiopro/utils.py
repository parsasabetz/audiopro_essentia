import numpy as np


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


def optimized_convert_to_native_types(data):
    """Convert numpy types to native Python types using vectorized operations"""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)
    elif isinstance(data, dict):
        return {
            key: optimized_convert_to_native_types(value) for key, value in data.items()
        }
    elif isinstance(data, list):
        return list(map(optimized_convert_to_native_types, data))
    return data
