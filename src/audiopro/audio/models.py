# Standard library imports
from typing import Dict, Optional, Union, List
from dataclasses import dataclass, field

# Local application imports
from audiopro.output.types import FrequencyBands

# Define a type alias for feature values
FeatureValue = Union[float, List[float], FrequencyBands]


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

    # Only time is required, all other fields are optional and only included if computed
    time: float
    _computed_features: Dict[str, FeatureValue] = field(default_factory=dict)

    def __post_init__(self):
        """Validate time is non-negative."""
        if self.time < 0:
            raise ValueError("Time must be non-negative")

    @classmethod
    def create(
        cls, time: float, **computed_features: Dict[str, FeatureValue]
    ) -> "FrameFeatures":
        """
        Create a FrameFeatures instance with only computed features.

        Args:
            time: Time of the frame in milliseconds
            **computed_features: Key-value pairs of computed features

        Returns:
            FrameFeatures instance with only the computed features
        """
        return cls(time=time, _computed_features=computed_features)

    def to_dict(self) -> Dict[str, FeatureValue]:
        """Convert to dictionary, including only computed features.

        Returns:
            Dict containing only the features that were computed
        """
        return {"time": self.time, **self._computed_features}

    def __getattr__(self, name: str) -> Optional[FeatureValue]:
        """Get a feature value, returning None if not computed."""
        if name.startswith("_"):
            raise AttributeError(f"No such attribute: {name}")
        return self._computed_features.get(name)
