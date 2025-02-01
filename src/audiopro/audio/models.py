# typing imports
from typing import Dict, List

# Standard library imports
from dataclasses import dataclass


@dataclass
class FrameFeatures:
    """Type-safe container for frame features.

    Attributes:
        time: Time of the frame in milliseconds
        rms: Root mean square value of the frame
        spectral_centroid: Spectral centroid of the frame
        spectral_bandwidth: Spectral bandwidth of the frame
        spectral_flatness: Spectral flatness of the frame
        spectral_rolloff: Spectral rolloff of the frame
        zero_crossing_rate: Zero crossing rate of the frame
        mfcc: Mel-frequency cepstral coefficients
        frequency_bands: Energy in different frequency bands
        chroma: Chroma vector of the frame
    """

    time: float
    rms: float
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_flatness: float
    spectral_rolloff: float
    zero_crossing_rate: float
    mfcc: List[float]
    frequency_bands: Dict[str, float]
    chroma: List[float]
