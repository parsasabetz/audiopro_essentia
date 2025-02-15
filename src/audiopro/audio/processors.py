# typing imports
from typing import Optional, Tuple

# Third-party imports
import numpy as np
from numpy.typing import NDArray
import torch

# Local models
from audiopro.audio.models import FrameFeatures

# Local util imports
from audiopro.utils.constants import (  # pylint: disable=no-name-in-module
    HOP_LENGTH,
    FRAME_LENGTH,
)
from audiopro.utils.logger import get_logger
from audiopro.output.types import AVAILABLE_FEATURES, SPECTRAL_FEATURES, FeatureConfig

# Local error handling imports
from audiopro.errors.exceptions import (
    FeatureExtractionError,
    AudioValidationError,
    SpectralFeatureError,
)
from audiopro.errors.tracking import error_tracking_context, ErrorStats

# Local audio imports
from audiopro.audio.transforms import get_transforms
from audiopro.audio.spectrogram import (
    compute_spectrogram,
    process_spectral_features,
)

# Setup logger
logger = get_logger(__name__)

# Pre-compute constants for efficiency
EPS: float = torch.finfo(torch.float32).eps
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_frame(
    frame_data: Tuple[int, NDArray[np.float32]],
    sample_rate: int,
    feature_config: Optional[FeatureConfig] = None,
    start_sample: int = 0,
) -> Tuple[int, Optional[FrameFeatures]]:
    """
    Process a single frame of audio data to extract various features.

    Parameters:
        - frame_data (Tuple[int, NDArray[np.float32]]): A tuple containing the frame index and the frame data.
        - sample_rate (int): The sample rate of the audio data.
        - feature_config (Optional[FeatureConfig]): Configuration for which features to extract.
        - start_sample (int): Sample offset from the start of the audio file.

    Returns:
        - Tuple[int, Optional[FrameFeatures]]: A tuple containing the frame index and the extracted features, or None if an error occurred.
    """
    frame_stats = ErrorStats()

    with error_tracking_context(frame_stats):
        try:
            frame_index, frame = frame_data

            # Determine which transforms are needed
            compute_spectrum = bool(
                SPECTRAL_FEATURES & (feature_config or AVAILABLE_FEATURES)
            )
            compute_mfcc = "mfcc" in (feature_config or AVAILABLE_FEATURES)

            # Get cached transforms for this sample rate
            _spectrum_transform, mfcc_transform = get_transforms(
                sample_rate, compute_spectrum, compute_mfcc
            )

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
                if len(frame) < FRAME_LENGTH:
                    frame = np.pad(frame, (0, FRAME_LENGTH - len(frame)))
                elif len(frame) > FRAME_LENGTH:
                    frame = frame[:FRAME_LENGTH]

                # Convert to tensor and move to device
                frame_tensor = torch.from_numpy(frame).to(DEVICE)
                frame_tensor = frame_tensor.view(1, 1, -1)

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
                if compute_spectrum:
                    # Create spectrogram and process spectral features
                    spec_tensor = compute_spectrogram(frame_tensor)
                    feature_values.update(
                        process_spectral_features(
                            spec_tensor,
                            sample_rate,
                            feature_config,
                            mfcc_transform,
                            frame_tensor,
                        )
                    )

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

            except (AudioValidationError, FeatureExtractionError, SpectralFeatureError):
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
