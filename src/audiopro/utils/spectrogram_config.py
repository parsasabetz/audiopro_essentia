# typing imports
from typing import Optional, Tuple

# third-party imports
import torch
import numpy as np


class SpectrogramConfig:
    """
    Configuration class for generating spectrograms.

    Attributes:
        DEFAULT_FFT_SIZE (int): Default size of the FFT window.
        DEFAULT_HOP_LENGTH (int): Default hop length for the STFT.
        EXPECTED_BINS (int): Expected number of frequency bins in the spectrogram.
        MIN_VALID_AMPLITUDE (float): Minimum valid amplitude for the spectrogram.

    Args:
        fft_size (int): Size of the FFT window. Must be a power of 2. Defaults to DEFAULT_FFT_SIZE.
        hop_length (int): Hop length for the STFT. Defaults to DEFAULT_HOP_LENGTH.

    Methods:
        validate_spectrogram_shape(spectrogram: np.ndarray) -> Tuple[bool, Optional[str]]:
            Validates the shape and amplitude of the given spectrogram.

        get_window(device: Optional[torch.device] = None) -> torch.Tensor:
            Returns the window function for STFT.

        get_default_config() -> "SpectrogramConfig":
            Returns a default instance of SpectrogramConfig.

        __repr__() -> str:
            Returns a string representation of the SpectrogramConfig instance.
    """

    # Ensure these match the constants in other files
    DEFAULT_FFT_SIZE = 2048
    DEFAULT_HOP_LENGTH = 512
    EXPECTED_BINS = 1025  # (FFT_SIZE // 2) + 1
    MIN_VALID_AMPLITUDE = 1e-10

    def __init__(
        self, fft_size: int = DEFAULT_FFT_SIZE, hop_length: int = DEFAULT_HOP_LENGTH
    ):
        if not self._is_power_of_two(fft_size):
            raise ValueError("FFT size must be a power of 2")
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.expected_bins = (fft_size // 2) + 1

    @staticmethod
    def _is_power_of_two(n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0

    def validate_spectrogram_shape(
        self, spectrogram: np.ndarray
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates the shape and amplitude of a given spectrogram.

        Args:
            spectrogram (np.ndarray): The spectrogram to validate.

        Returns:
            Tuple[bool, Optional[str]]: A tuple where the first element is a boolean indicating
            whether the spectrogram is valid, and the second element is an optional string
            containing an error message if the spectrogram is not valid.
        """

        if spectrogram.size == 0:
            return False, "Empty spectrogram"

        if spectrogram.shape[0] != self.expected_bins:
            return (
                False,
                f"Unexpected spectrogram shape: got {spectrogram.shape[0]} bins, expected {self.expected_bins}",
            )

        # Check for valid amplitude range
        if np.max(np.abs(spectrogram)) < self.MIN_VALID_AMPLITUDE:
            return False, "Spectrogram has insufficient amplitude"

        return True, None

    def get_window(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Generates a Hann window tensor for the given FFT size.

        Args:
            device (Optional[torch.device]): The device on which to create the Hann window tensor.
                                             If None, the tensor will be created on the default device.

        Returns:
            torch.Tensor: A tensor containing the Hann window.
        """

        return torch.hann_window(self.fft_size, device=device)

    @staticmethod
    def get_default_config() -> "SpectrogramConfig":
        return SpectrogramConfig()

    def __repr__(self) -> str:
        return (
            f"SpectrogramConfig(fft_size={self.fft_size}, hop_length={self.hop_length})"
        )
