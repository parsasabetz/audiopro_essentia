# typing imports
from typing import Dict, Optional, Tuple

# Standard library imports
from functools import lru_cache

# Third-party imports
import numpy as np
from numpy.typing import NDArray
import torch
import torchaudio.transforms as T

# Local application imports
from audiopro.audio.models import FrameFeatures
from audiopro.utils.constants import (  # pylint: disable=no-name-in-module
    HOP_LENGTH,
    FRAME_LENGTH,
    FREQUENCY_BANDS,
)
from audiopro.utils.logger import get_logger
from audiopro.output.types import AVAILABLE_FEATURES, SPECTRAL_FEATURES, FeatureConfig
from audiopro.errors.exceptions import (
    FeatureExtractionError,
    AudioValidationError,
    SpectralFeatureError,
)
from audiopro.errors.tracking import error_tracking_context, ErrorStats

# Setup logger
logger = get_logger(__name__)

# Pre-compute constants for efficiency
EPS: float = torch.finfo(torch.float32).eps
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FFT size must be power of 2 for efficiency and correct frequency resolution
N_FFT = 2048  # This will give us 1025 frequency bins (N_FFT//2 + 1)
WINDOW_FN = torch.hann_window(N_FFT, device=DEVICE)  # Use N_FFT for window size


@lru_cache(maxsize=8)
def get_transforms(sample_rate: int):
    """Get all audio transforms for a given sample rate."""
    spectrum_transform = T.Spectrogram(
        n_fft=N_FFT,
        win_length=N_FFT,  # Match window length with FFT size
        hop_length=HOP_LENGTH,
        pad=0,
        window_fn=torch.hann_window,  # Use default hann window
        power=2.0,  # Use power spectrum
        normalized=False,
        wkwargs=None,
        center=True,
        pad_mode="reflect",
        onesided=True,
    ).to(DEVICE)

    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,
        melkwargs={
            "n_fft": N_FFT,
            "win_length": N_FFT,  # Match window length with FFT size
            "hop_length": HOP_LENGTH,
            "n_mels": 40,
            "mel_scale": "htk",
            "normalized": True,
            "center": True,
            "pad_mode": "reflect",
            "power": 2.0,
            "window_fn": torch.hann_window,  # Use default hann window
        },
    ).to(DEVICE)

    return spectrum_transform, mfcc_transform


@lru_cache(maxsize=32)
def get_frequency_bins(sample_rate: int) -> NDArray[np.float32]:
    """
    Cached computation of frequency bins.

    Args:
        sample_rate: The sample rate of the audio

    Returns:
        Array of frequency bins
    """
    return np.linspace(
        0, sample_rate / 2, N_FFT // 2 + 1, dtype=np.float32
    )  # Use N_FFT instead of FRAME_LENGTH


def compute_frequency_bands(
    spec: NDArray[np.float32], sample_rate: int
) -> Dict[str, float]:
    """
    Compute the average magnitude of the spectrogram within predefined frequency bands.

    Args:
        spec: The input spectrogram array (power spectrum)
        sample_rate: The sample rate of the audio signal

    Returns:
        Dictionary mapping frequency band names to their average magnitudes
    """
    logger.debug(f"Input spec shape: {spec.shape}")

    # Ensure spec is 1D and handle empty or invalid input
    if spec.size == 0:
        logger.error("Empty spectrogram received")
        return {band: 0.0 for band in FREQUENCY_BANDS}

    # Get frequency bins
    freqs = get_frequency_bins(sample_rate)
    logger.debug(f"Frequency bins shape: {freqs.shape}")

    # Ensure spec has the correct shape
    if spec.shape[0] != freqs.shape[0]:
        logger.error(f"Shape mismatch: spec {spec.shape}, freqs {freqs.shape}")
        if spec.shape[0] < freqs.shape[0]:
            # Pad with zeros if spectrogram is too short
            spec = np.pad(spec, (0, freqs.shape[0] - spec.shape[0]))
        else:
            # Truncate if spectrogram is too long
            spec = spec[: freqs.shape[0]]
        logger.debug(f"Adjusted spec shape: {spec.shape}")

    result = {}

    # Apply log scaling to better handle the dynamic range
    spec_db = 10 * np.log10(np.maximum(spec, EPS))

    # Normalize to dB scale
    spec_db = np.clip(spec_db, -80, 0)  # Clip to -80 dB
    spec_db = spec_db + 80  # Shift to positive range
    spec_db = spec_db / 80  # Normalize to [0, 1]

    for band_name, (low, high) in FREQUENCY_BANDS.items():
        low_idx = np.searchsorted(freqs, low, side="left")
        high_idx = np.searchsorted(freqs, high, side="right")

        # Ensure valid indices
        low_idx = min(low_idx, len(freqs) - 1)
        high_idx = min(high_idx, len(freqs))

        if high_idx > low_idx:
            try:
                # Calculate band energy
                band_magnitudes = spec_db[low_idx:high_idx]
                band_freqs = freqs[low_idx:high_idx]

                # Use frequency-weighted average
                weights = np.log10(band_freqs / (low + EPS) + 1)
                weights = weights / (
                    np.sum(weights) + EPS
                )  # Normalize weights to sum to 1

                band_energy = np.sum(band_magnitudes * weights)
                result[band_name] = float(band_energy)
            except (ValueError, IndexError, TypeError) as e:
                logger.error(f"Error processing band {band_name}: {str(e)}")
                result[band_name] = 0.0
        else:
            result[band_name] = 0.0

    # Normalize the results to sum to 1
    total_energy = sum(result.values()) + EPS
    result = {k: v / total_energy for k, v in result.items()}

    return result


def process_frame(
    frame_data: Tuple[int, NDArray[np.float32]],
    sample_rate: int,
    frame_length: int,
    feature_config: Optional[FeatureConfig] = None,
    start_sample: int = 0,
) -> Tuple[int, Optional[FrameFeatures]]:
    """
    Process a single frame of audio data to extract various features.

    Parameters:
        - frame_data (Tuple[int, NDArray[np.float32]]): A tuple containing the frame index and the frame data.
        - sample_rate (int): The sample rate of the audio data.
        - frame_length (int): The length of the frame.
        - feature_config (Optional[FeatureConfig]): Configuration for which features to extract. If None, all available features are extracted.
        - start_sample (int): Sample offset from the start of the audio file (default: 0)

    Returns:
        - Tuple[int, Optional[FrameFeatures]]: A tuple containing the frame index and the extracted features, or None if an error occurred.

    Raises:
        - AudioValidationError: If the frame data is invalid.
        - FeatureExtractionError: If there is an error during feature extraction.
        - SpectralFeatureError: If there is an error during spectral feature extraction.
        - Exception: For any other unexpected errors.

    Features extracted may include:
        - volume
        - rms
        - spectral_centroid
        - spectral_bandwidth
        - spectral_flatness
        - spectral_rolloff
        - mfcc
        - frequency_bands
        - zero_crossing_rate
    """
    frame_stats = ErrorStats()

    with error_tracking_context(frame_stats):
        try:
            frame_index, frame = frame_data

            # Get cached transforms for this sample rate
            spectrum_transform, mfcc_transform = get_transforms(sample_rate)

            try:
                # Basic validation
                if (
                    not isinstance(frame, np.ndarray)
                    or frame.size == 0
                    or not np.isfinite(frame).all()
                ):
                    raise AudioValidationError("Invalid frame data")

                # Convert to float32 and mono if needed
                frame = frame.astype(np.float32, copy=False)
                if frame.ndim > 1:
                    frame = np.mean(frame, axis=1)

                # Pad or truncate to match FFT size
                if len(frame) < N_FFT:
                    frame = np.pad(frame, (0, N_FFT - len(frame)))
                elif len(frame) > N_FFT:
                    frame = frame[:N_FFT]

                # Convert to tensor and move to device
                frame_tensor = torch.from_numpy(frame).to(DEVICE)
                frame_tensor = frame_tensor.view(
                    1, 1, -1
                )  # Add batch and channel dimensions

                # Initialize features
                feature_values = {}

                # Compute basic features
                rms_tensor = torch.sqrt(torch.mean(frame_tensor.pow(2)))

                if "volume" in (feature_config or AVAILABLE_FEATURES):
                    feature_values["volume"] = float(
                        20.0 * torch.log10(rms_tensor + EPS)
                    )

                if "rms" in (feature_config or AVAILABLE_FEATURES):
                    feature_values["rms"] = float(rms_tensor)

                # Compute spectral features if needed
                if bool(SPECTRAL_FEATURES & (feature_config or AVAILABLE_FEATURES)):
                    # Compute spectrogram and ensure correct shape
                    logger.debug(
                        f"Input frame tensor shape before transform: {frame_tensor.shape}"
                    )

                    # Create spectrogram transform with explicit parameters and ensure window matches FFT size
                    window = torch.hann_window(N_FFT, device=DEVICE)
                    spec_tensor = torch.stft(
                        frame_tensor.squeeze(),  # Remove batch and channel dimensions
                        n_fft=N_FFT,
                        hop_length=HOP_LENGTH,
                        win_length=N_FFT,
                        window=window,
                        center=True,
                        normalized=False,
                        onesided=True,
                        return_complex=True,
                    )

                    # Convert complex STFT to power spectrogram
                    spec_tensor = torch.abs(spec_tensor).pow(2)

                    logger.debug(
                        f"Raw spectrogram shape after STFT: {spec_tensor.shape}"
                    )

                    # Extract the frequency dimension correctly
                    if (
                        spec_tensor.dim() == 3
                    ):  # (freq, time, 2) or (freq, time, real/imag)
                        spec_tensor = spec_tensor[..., 0]  # Take first time frame
                    elif spec_tensor.dim() == 2:  # (freq, time)
                        spec_tensor = spec_tensor[:, 0]  # Take first time frame

                    # Ensure frequency dimension is correct (N_FFT//2 + 1)
                    expected_bins = N_FFT // 2 + 1
                    if spec_tensor.shape[0] != expected_bins:
                        logger.error(
                            f"Unexpected frequency bins: got {spec_tensor.shape[0]}, expected {expected_bins}"
                        )
                        if spec_tensor.shape[0] < expected_bins:
                            spec_tensor = torch.nn.functional.pad(
                                spec_tensor.unsqueeze(
                                    0
                                ),  # Add batch dimension for padding
                                (0, expected_bins - spec_tensor.shape[0]),
                                mode="constant",
                            ).squeeze(0)
                        else:
                            spec_tensor = spec_tensor[:expected_bins]

                    logger.debug(f"Final spectrogram shape: {spec_tensor.shape}")

                    # Update frequency tensor to match spectrogram shape
                    freq_tensor = torch.linspace(
                        0, sample_rate / 2, expected_bins, device=DEVICE
                    )

                    # Handle the case where spectrogram has no valid data
                    if torch.all(spec_tensor == 0):
                        raise SpectralFeatureError(
                            "Zero spectrum detected", feature_name="spectral_features"
                        )

                    # Ensure spec_tensor and freq_tensor have compatible shapes for operations
                    spec_tensor = spec_tensor.view(-1)  # Flatten to 1D
                    freq_tensor = freq_tensor.view(-1)  # Flatten to 1D

                    # Compute spectral features efficiently
                    if "spectral_centroid" in (feature_config or AVAILABLE_FEATURES):
                        spec_sum = torch.sum(spec_tensor)
                        centroid = float(
                            torch.sum(freq_tensor * spec_tensor) / (spec_sum + EPS)
                        )
                        feature_values["spectral_centroid"] = centroid

                        if "spectral_bandwidth" in (
                            feature_config or AVAILABLE_FEATURES
                        ):
                            # Reuse centroid and spec_sum
                            bandwidth = float(
                                torch.sqrt(
                                    torch.sum(
                                        (freq_tensor - centroid).pow(2) * spec_tensor
                                    )
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
                            # Convert spectrogram to numpy and ensure correct shape
                            spec_np = spec_tensor.cpu().numpy()
                            logger.debug(
                                f"Spectrogram shape before conversion: {spec_np.shape}"
                            )

                            # Ensure we're using the full frequency spectrum
                            if spec_np.shape[-1] != FRAME_LENGTH // 2 + 1:
                                logger.error(
                                    f"Unexpected spectrogram shape: {spec_np.shape}, expected last dimension to be {FRAME_LENGTH // 2 + 1}"
                                )
                                # Reshape or pad the spectrogram to match frequency bins
                                if spec_np.ndim == 1:
                                    spec_np = np.pad(
                                        spec_np,
                                        (0, (FRAME_LENGTH // 2 + 1) - spec_np.shape[0]),
                                    )
                                else:
                                    spec_np = spec_np[..., : FRAME_LENGTH // 2 + 1]

                            # Handle different spectrogram shapes
                            if spec_np.ndim == 3:  # (batch, channel, freq)
                                spec_np = spec_np.squeeze(0).squeeze(0)
                            elif spec_np.ndim == 2:  # (channel, freq)
                                spec_np = spec_np.squeeze(0)

                            logger.debug(
                                f"Spectrogram shape after conversion: {spec_np.shape}"
                            )

                            feature_values["frequency_bands"] = compute_frequency_bands(
                                spec_np, sample_rate
                            )
                        except Exception as e:
                            logger.error(f"Failed to compute frequency bands: {str(e)}")
                            feature_values["frequency_bands"] = {
                                band: 0.0 for band in FREQUENCY_BANDS
                            }

                if "zero_crossing_rate" in (feature_config or AVAILABLE_FEATURES):
                    feature_values["zero_crossing_rate"] = float(
                        torch.mean(
                            torch.abs(
                                torch.sign(frame_tensor[..., 1:])
                                - torch.sign(frame_tensor[..., :-1])
                            )
                        )
                        / 2.0
                    )

                # Calculate correct time by including the start_sample offset
                absolute_sample = start_sample + (frame_index * HOP_LENGTH)
                time_ms = (absolute_sample / sample_rate) * 1000
                return frame_index, FrameFeatures.create(time=time_ms, **feature_values)

            except (
                AudioValidationError,
                FeatureExtractionError,
                SpectralFeatureError,
            ):
                raise
            except Exception as e:
                raise FeatureExtractionError(f"Error processing frame: {str(e)}") from e

        except (
            AudioValidationError,
            FeatureExtractionError,
            SpectralFeatureError,
        ) as e:
            logger.error("Frame processing failed: %s", str(e))
            return frame_index, None
