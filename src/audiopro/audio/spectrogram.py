# typing imports
from typing import Dict, Optional

# Third-party imports
import torch
import numpy as np
from functools import lru_cache

# Local util imports
from audiopro.utils.constants import (  # pylint: disable=no-name-in-module
    FRAME_LENGTH,
    HOP_LENGTH,
    FREQUENCY_BANDS,
)
from audiopro.utils.logger import get_logger

# Other local imports
from audiopro.errors.exceptions import SpectralFeatureError
from audiopro.output.types import FeatureConfig
from audiopro.output.feature_flags import FeatureFlagSet, create_feature_flags
from audiopro.audio.frequency import compute_frequency_bands

# Setup logger
logger = get_logger(__name__)

# Pre-compute constants for efficiency
EPS: float = torch.finfo(torch.float32).eps
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_spectrogram(frame_tensor: torch.Tensor) -> torch.Tensor:
    """Computes the spectrogram of a given audio frame tensor.

    This function takes an audio frame tensor as input, applies a Short-Time Fourier Transform (STFT)
    to convert it into the frequency domain, and then computes the power spectrogram.
    The spectrogram is then normalized.

    Args:
        frame_tensor (torch.Tensor): A tensor representing the audio frame.
            Expected shape is (1, frame_length) where frame_length is the length of the audio frame.

    Returns:
        torch.Tensor: A tensor representing the normalized spectrogram of the audio frame.
            The shape of the output tensor is (num_freqs, num_frames), where num_freqs is the number
            of frequency bins and num_frames is the number of time frames.
    """

    logger.debug(f"Input frame tensor shape before transform: {frame_tensor.shape}")

    window = torch.hann_window(FRAME_LENGTH, device=DEVICE)
    spec_tensor = torch.stft(
        frame_tensor.squeeze(),
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        win_length=FRAME_LENGTH,
        window=window,
        center=True,
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    # Convert complex STFT to power spectrogram
    spec_tensor = torch.abs(spec_tensor).pow(2)
    logger.debug(f"Raw spectrogram shape after STFT: {spec_tensor.shape}")

    # Extract and normalize frequency dimension
    spec_tensor = normalize_spectrogram(spec_tensor)

    return spec_tensor


def normalize_spectrogram(spec_tensor: torch.Tensor) -> torch.Tensor:
    """Normalizes a spectrogram tensor to a standard format.

    This function takes a spectrogram tensor as input and normalizes it to ensure
    it has the expected dimensions. Specifically, it ensures that the tensor has
    the correct number of frequency bins based on the FRAME_LENGTH constant.

    Args:
        spec_tensor (torch.Tensor): The input spectrogram tensor. It can be
            either 2D (freq, time) or 3D (freq, time, channels).

    Returns:
        torch.Tensor: The normalized spectrogram tensor.

    Raises:
        Logs an error message if the number of frequency bins in the input tensor
        does not match the expected number based on FRAME_LENGTH.  If the number
        of bins is less than expected, the tensor is padded. If it's greater,
        the tensor is truncated.
    """
    # Extract the frequency dimension correctly
    if spec_tensor.dim() == 3:  # (freq, time, 2) or (freq, time, real/imag)
        spec_tensor = spec_tensor[..., 0]  # Take first time frame
    elif spec_tensor.dim() == 2:  # (freq, time)
        spec_tensor = spec_tensor[:, 0]  # Take first time frame

    # Ensure frequency dimension is correct
    expected_bins = FRAME_LENGTH // 2 + 1
    if spec_tensor.shape[0] != expected_bins:
        logger.error(
            f"Unexpected frequency bins: got {spec_tensor.shape[0]}, expected {expected_bins}"
        )
        if spec_tensor.shape[0] < expected_bins:
            spec_tensor = torch.nn.functional.pad(
                spec_tensor.unsqueeze(0),
                (0, expected_bins - spec_tensor.shape[0]),
                mode="constant",
            ).squeeze(0)
        else:
            spec_tensor = spec_tensor[:expected_bins]

    return spec_tensor


@lru_cache(maxsize=16)
def get_pitch_class_mapping(
    sample_rate: int, bins_per_octave: int = 12, f_ref: float = 55.0
) -> torch.Tensor:
    """
    Compute a mapping from frequency bins to pitch classes.
    This function precomputes a mapping from frequency bins to pitch classes for a given sample rate,
    number of bins per octave, and reference frequency. The mapping is cached to improve performance
    for repeated calls with the same parameters.
    Args:
        sample_rate (int): The sample rate of the audio signal.
        bins_per_octave (int, optional): The number of bins per octave. Default is 12.
        f_ref (float, optional): The reference frequency in Hz. Default is 55.0 Hz.
    Returns:
        torch.Tensor: A tensor containing the pitch class mapping for each frequency bin.
    """

    # Precompute frequency vector from 0 to sample_rate/2
    freq = torch.linspace(0, sample_rate / 2, FRAME_LENGTH // 2 + 1, device=DEVICE)

    # Mark indices with zero frequency as invalid (-1)
    mapping = torch.full((freq.shape[0],), -1, device=DEVICE, dtype=torch.long)

    nonzero = freq > 0
    freq_nonzero = freq[nonzero]

    pitch_class = torch.remainder(
        torch.round(bins_per_octave * torch.log2(freq_nonzero / f_ref)), bins_per_octave
    ).to(torch.long)
    mapping[nonzero] = pitch_class
    return mapping


def process_spectral_features(
    spec_tensor: torch.Tensor,
    sample_rate: int,
    feature_config: Optional[FeatureConfig],
    mfcc_transform: torch.nn.Module,
    frame_tensor: torch.Tensor,
) -> Dict[str, float]:
    """Compute spectral features from a spectrogram.

    Args:
        spec_tensor (torch.Tensor): Spectrogram tensor.
        sample_rate (int): Sample rate of the audio.
        feature_config (Optional[FeatureConfig]): Configuration for feature extraction.
        mfcc_transform (torch.nn.Module): MFCC transformation module.
        frame_tensor (torch.Tensor): The original audio frame tensor.

    Returns:
        Dict[str, float]: Dictionary containing computed spectral features.

    Raises:
        SpectralFeatureError: If the spectrogram contains only zeros.
    """

    feature_values = {}
    logger.debug(f"Final spectrogram shape: {spec_tensor.shape}")

    # Create feature flags once for efficient checking
    feature_flags = create_feature_flags(feature_config)

    # Update frequency tensor to match spectrogram shape
    freq_tensor = torch.linspace(
        0, sample_rate / 2, FRAME_LENGTH // 2 + 1, device=DEVICE
    )

    # Handle the case where spectrogram has no valid data
    if torch.all(spec_tensor == 0):
        raise SpectralFeatureError(
            "Zero spectrum detected", feature_name="spectral_features"
        )

    # Ensure shapes are compatible
    spec_tensor = spec_tensor.view(-1)
    freq_tensor = freq_tensor.view(-1)

    # Compute spectral features using efficient bit operations
    if feature_flags.is_enabled("spectral_centroid"):
        spec_sum = torch.sum(spec_tensor)
        centroid = float(torch.sum(freq_tensor * spec_tensor) / (spec_sum + EPS))
        feature_values["spectral_centroid"] = centroid

        if feature_flags.is_enabled("spectral_bandwidth"):
            bandwidth = float(
                torch.sqrt(
                    torch.sum((freq_tensor - centroid).pow(2) * spec_tensor)
                    / (spec_sum + EPS)
                )
            )
            feature_values["spectral_bandwidth"] = bandwidth

    if feature_flags.is_enabled("spectral_flatness"):
        feature_values["spectral_flatness"] = float(
            torch.exp(torch.mean(torch.log(spec_tensor + EPS)))
            / (torch.mean(spec_tensor) + EPS)
        )

    if feature_flags.is_enabled("spectral_rolloff"):
        cumsum = torch.cumsum(spec_tensor, dim=0)
        threshold = 0.85 * torch.sum(spec_tensor)
        rolloff_idx = torch.where(cumsum >= threshold)[0][0]
        feature_values["spectral_rolloff"] = float(
            freq_tensor[min(rolloff_idx, freq_tensor.shape[0] - 1)]
        )

    # Optimized vectorized chroma computation using cached mapping
    if feature_flags.is_enabled("chroma"):
        try:
            bins_per_octave = 12
            mapping = get_pitch_class_mapping(sample_rate, bins_per_octave, 55.0)
            valid_mask = mapping >= 0
            # Use the matching indices from spec_tensor (assumed aligned with mapping)
            spec_nonzero = spec_tensor[valid_mask]
            pitch_classes = mapping[valid_mask]
            chroma_vector = torch.zeros(bins_per_octave, device=DEVICE)
            chroma_vector = chroma_vector.scatter_add_(0, pitch_classes, spec_nonzero)
            feature_values["chroma"] = chroma_vector.cpu().tolist()
        except Exception as e:
            logger.error(f"Chroma computation failed: {str(e)}")
            feature_values["chroma"] = [0.0] * 12

    process_mfcc_and_frequency_bands(
        feature_values,
        feature_flags,
        mfcc_transform,
        frame_tensor,
        spec_tensor,
        sample_rate,
    )

    return feature_values


def process_mfcc_and_frequency_bands(
    feature_values: Dict[str, float],
    feature_flags: FeatureFlagSet,
    mfcc_transform: torch.nn.Module,
    frame_tensor: torch.Tensor,
    spec_tensor: torch.Tensor,
    sample_rate: int,
) -> None:
    """Processes audio frames to extract MFCC coefficients and frequency bands,
    and stores them in the feature_values dictionary.

    Args:
        feature_values (Dict[str, float]): A dictionary to store the computed features.
        feature_flags (FeatureFlagSet): Efficient feature flag set for checking enabled features.
        mfcc_transform (torch.nn.Module): A PyTorch module for computing MFCCs.
        frame_tensor (torch.Tensor): A PyTorch tensor representing the audio frame.
        spec_tensor (torch.Tensor): A PyTorch tensor representing the spectrogram of the audio frame.
        sample_rate (int): The sample rate of the audio.
    """

    if feature_flags.is_enabled("mfcc"):
        try:
            mfcc_coeffs = mfcc_transform(frame_tensor).squeeze()
            if mfcc_coeffs.dim() == 2:
                mfcc_coeffs = mfcc_coeffs[:, 0]
            mfcc_coeffs = mfcc_coeffs.cpu().numpy()
            feature_values["mfcc"] = (
                mfcc_coeffs[:13]
                if len(mfcc_coeffs) >= 13
                else np.pad(mfcc_coeffs, (0, 13 - len(mfcc_coeffs)))
            ).tolist()
        except (RuntimeError, ValueError) as e:
            logger.error(f"MFCC computation failed: {str(e)}")
            feature_values["mfcc"] = [0.0] * 13

    if feature_flags.is_enabled("frequency_bands"):
        try:
            spec_np = spec_tensor.cpu().numpy()
            feature_values["frequency_bands"] = compute_frequency_bands(
                spec_np, sample_rate
            )
        except (RuntimeError, ValueError) as e:
            logger.error(f"Failed to compute frequency bands: {str(e)}")
            feature_values["frequency_bands"] = {band: 0.0 for band in FREQUENCY_BANDS}
