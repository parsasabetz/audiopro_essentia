"""Feature flag management using efficient bit operations."""

# typing imports
from typing import Optional

# local imports
from .types import FeatureConfig, FEATURE_NAMES, AVAILABLE_FEATURES, SPECTRAL_FEATURES


class FeatureFlagSet:
    """
    FeatureFlagSet is a class that manages feature flags using bit operations.

    This class provides an efficient way to track and check which audio features
    are enabled using bit operations instead of dictionary lookups. Each feature
    is represented by a bit in an integer flag, allowing for fast checks and
    minimal memory usage.

    Attributes:
        _flags (int): A bit mask representing the enabled features.
        _feature_indices (dict): A mapping from feature names to their respective indices.

    Methods:
        __init__(feature_config: Optional[FeatureConfig] = None):
            Initializes the feature flags from the given configuration.

        is_enabled(feature_name: str) -> bool:
            Checks if a specific feature is enabled using bit operations.

        get_enabled_features() -> frozenset:
            Returns a frozenset of all enabled features using bit operations.

        has_spectral_features() -> bool:
            Checks if any spectral features are enabled.

    Example:
        >>> feature_config = {"rms": True, "volume": True}
        >>> flags = FeatureFlagSet(feature_config)
        >>> flags.is_enabled("rms")
        True
        >>> flags.is_enabled("mfcc")
        False
    """

    def __init__(self, feature_config: Optional[FeatureConfig] = None):
        """
        Initialize feature flags from configuration.

        Creates a bit mask where each bit represents whether a feature is enabled (1)
        or disabled (0). If no configuration is provided, all features are enabled
        by default.

        Args:
            feature_config (Optional[FeatureConfig]): Configuration dictionary mapping
                feature names to boolean values. If None, all features are enabled.
        """
        # Create a bit mask where 1 means enabled
        self._flags = 0
        if feature_config is None:
            # All features enabled
            self._flags = (1 << len(FEATURE_NAMES)) - 1
        else:
            for i, feature in enumerate(FEATURE_NAMES):
                if feature_config.get(feature, False):
                    self._flags |= 1 << i

        # Cache the feature name to index mapping
        self._feature_indices = {name: i for i, name in enumerate(FEATURE_NAMES)}

    def is_enabled(self, feature_name: str) -> bool:
        """
        Check if a specific feature is enabled.

        Uses bit operations to efficiently check if a feature is enabled.

        Args:
            feature_name (str): The name of the feature to check.

        Returns:
            bool: True if the feature is enabled, False otherwise.
        """
        if self._flags == (1 << len(FEATURE_NAMES)) - 1:  # All features enabled
            return True
        idx = self._feature_indices.get(feature_name)
        if idx is None:
            return False
        return bool(self._flags & (1 << idx))

    def get_enabled_features(self) -> frozenset:
        """
        Get the set of enabled features.

        Uses bit operations to efficiently determine which features are enabled.

        Returns:
            frozenset: A set of feature names that are currently enabled.
            If all features are enabled, returns the set of all available features.
        """
        if self._flags == (1 << len(FEATURE_NAMES)) - 1:  # All features enabled
            return AVAILABLE_FEATURES
        return frozenset(
            name
            for name, idx in self._feature_indices.items()
            if self._flags & (1 << idx)
        )

    @property
    def has_spectral_features(self) -> bool:
        """
        Check if any spectral features are enabled.

        Uses a pre-computed bit mask to efficiently check if any spectral features
        are enabled in the current configuration.

        Returns:
            bool: True if any spectral features are enabled, False otherwise.
        """
        spectral_mask = sum(1 << self._feature_indices[f] for f in SPECTRAL_FEATURES)
        return bool(self._flags & spectral_mask)


def create_feature_flags(
    feature_config: Optional[FeatureConfig] = None,
) -> FeatureFlagSet:
    """
    Create a feature flag set from a configuration.

    Factory function to create a new FeatureFlagSet instance with the given configuration.

    Args:
        feature_config (Optional[FeatureConfig]): The configuration for the features.
            If None, default configuration will be used (all features enabled).

    Returns:
        FeatureFlagSet: A new feature flag set instance.
    """
    return FeatureFlagSet(feature_config)
