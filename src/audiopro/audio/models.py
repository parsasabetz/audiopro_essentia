# typing imports
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


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
    rms: Optional[float] = None
    spectral_centroid: Optional[float] = None
    spectral_bandwidth: Optional[float] = None
    spectral_flatness: Optional[float] = None
    spectral_rolloff: Optional[float] = None
    zero_crossing_rate: Optional[float] = None
    mfcc: Optional[List[float]] = None
    frequency_bands: Optional[Dict[str, float]] = None
    chroma: Optional[List[float]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding None values.

        Returns:
            Dict containing only the features that were computed (non-None values)
        """
        # Start with time which is always included
        result = {"time": self.time}

        # Add other features only if they're not None
        for key, value in asdict(self).items():
            if key != "time" and value is not None:
                result[key] = value

        return result
