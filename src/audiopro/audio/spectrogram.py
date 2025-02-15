# typing imports
from typing import Dict, Optional

# Third-party imports
import torch
import numpy as np

# Local util imports
from audiopro.utils.constants import (  # pylint: disable=no-name-in-module
    FRAME_LENGTH,
    HOP_LENGTH,
    FREQUENCY_BANDS,
)
from audiopro.utils.logger import get_logger

# Other local imports
from audiopro.errors.exceptions import SpectralFeatureError
from audiopro.output.types import AVAILABLE_FEATURES, FeatureConfig
from audiopro.audio.frequency import compute_frequency_bands

# Setup logger
logger = get_logger(__name__)

# Pre-compute constants for efficiency
EPS: float = torch.finfo(torch.float32).eps
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_spectrogram(frame_tensor: torch.Tensor) -> torch.Tensor:
    """Compute spectrogram from frame tensor."""
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
    """Normalize spectrogram shape and dimensions."""
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


def process_spectral_features(
    spec_tensor: torch.Tensor,
    sample_rate: int,
    feature_config: Optional[FeatureConfig],
    mfcc_transform: torch.nn.Module,
    frame_tensor: torch.Tensor,
) -> Dict[str, float]:
    """Process spectral features from spectrogram."""
    feature_values = {}
    logger.debug(f"Final spectrogram shape: {spec_tensor.shape}")

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

    # Compute spectral features
    if "spectral_centroid" in (feature_config or AVAILABLE_FEATURES):
        spec_sum = torch.sum(spec_tensor)
        centroid = float(torch.sum(freq_tensor * spec_tensor) / (spec_sum + EPS))
        feature_values["spectral_centroid"] = centroid

        if "spectral_bandwidth" in (feature_config or AVAILABLE_FEATURES):
            bandwidth = float(
                torch.sqrt(
                    torch.sum((freq_tensor - centroid).pow(2) * spec_tensor)
                    / (spec_sum + EPS)
                )
            )
            feature_values["spectral_bandwidth"] = bandwidth

    if "spectral_flatness" in (feature_config or AVAILABLE_FEATURES):
        feature_values["spectral_flatness"] = float(
            torch.exp(torch.mean(torch.log(spec_tensor + EPS)))
            / (torch.mean(spec_tensor) + EPS)
        )

    if "spectral_rolloff" in (feature_config or AVAILABLE_FEATURES):
        cumsum = torch.cumsum(spec_tensor, dim=0)
        threshold = 0.85 * torch.sum(spec_tensor)
        rolloff_idx = torch.where(cumsum >= threshold)[0][0]
        feature_values["spectral_rolloff"] = float(
            freq_tensor[min(rolloff_idx, freq_tensor.shape[0] - 1)]
        )

    process_mfcc_and_frequency_bands(
        feature_values,
        feature_config,
        mfcc_transform,
        frame_tensor,
        spec_tensor,
        sample_rate,
    )

    return feature_values


def process_mfcc_and_frequency_bands(
    feature_values: Dict[str, float],
    feature_config: Optional[FeatureConfig],
    mfcc_transform: torch.nn.Module,
    frame_tensor: torch.Tensor,
    spec_tensor: torch.Tensor,
    sample_rate: int,
) -> None:
    """Process MFCC and frequency bands features."""
    if "mfcc" in (feature_config or AVAILABLE_FEATURES):
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

    if "frequency_bands" in (feature_config or AVAILABLE_FEATURES):
        try:
            spec_np = spec_tensor.cpu().numpy()
            feature_values["frequency_bands"] = compute_frequency_bands(
                spec_np, sample_rate
            )
        except (RuntimeError, ValueError) as e:
            logger.error(f"Failed to compute frequency bands: {str(e)}")
            feature_values["frequency_bands"] = {band: 0.0 for band in FREQUENCY_BANDS}
