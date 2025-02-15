"""
Module for audio feature extraction.
Features are extracted per frame with multiprocessing, preserving output order.
"""

# Typing imports
from typing import Optional, List, Iterator, Tuple, Callable, Dict

# Standard library imports
from functools import partial
import multiprocessing as mp
from itertools import islice
import gc

# Third-party imports
import numpy as np
from numpy.typing import NDArray

# Local application imports
from audiopro.utils.logger import get_logger
from audiopro.utils import (
    calculate_max_workers,
    FRAME_LENGTH,
    HOP_LENGTH,
    BATCH_SIZE,
)
from audiopro.output.types import FeatureConfig
from audiopro.errors.exceptions import (
    AudioProcessingError,
    ExtractionPipelineError,
    AudioValidationError,
)
from audiopro.errors.tracking import (
    ErrorStats,
    ErrorRateLimiter,
    error_tracking_context,
)
from .processors import process_frame
from .models import FrameFeatures

# Set up logger
logger = get_logger()


def create_frame_generator(
    audio_data: NDArray[np.float32], total_samples: int, total_frames: int
) -> Iterator[Tuple[int, NDArray[np.float32]]]:
    """
    Creates a memory-efficient generator for audio frames.

    Args:
        audio_data: The input audio data
        total_samples: Total number of samples in audio
        total_frames: Total number of frames to process

    Yields:
        Tuples of (frame_index, frame_data)
    """
    for frame_idx in range(total_frames):
        start_idx = frame_idx * HOP_LENGTH
        if (start_idx + FRAME_LENGTH) > total_samples:
            break
        frame_data = audio_data[start_idx : start_idx + FRAME_LENGTH].copy()
        if len(frame_data) == FRAME_LENGTH:
            yield (frame_idx, frame_data)
        # Explicitly delete frame data to help garbage collection
        del frame_data


def extract_features(
    audio_data: NDArray[np.float32],
    sample_rate: int,
    channels: int,
    on_feature: Optional[Callable[[FrameFeatures], None]] = None,
    feature_config: Optional[FeatureConfig] = None,
    start_sample: int = 0,  # Add start_sample parameter with default value
) -> List[Dict]:
    """
    Extracts features from the given audio data by processing it in frames.

    The audio data is divided into overlapping frames according to FRAME_LENGTH and HOP_LENGTH.
    Each frame is preprocessed with a window function and transformed using FFT to extract spectral features.
    Multiprocessing is used to distribute frame processing across multiple processes, ensuring a low CPU and memory footprint.

    Args:
        audio_data: The audio data as a numpy array
        sample_rate: The sample rate of the audio
        channels: Number of audio channels in the data
        on_feature: Optional callback for immediate feature processing
        feature_config: Optional configuration specifying which features to compute.
                      If None, all features will be computed.
        start_sample: Sample offset from the start of the audio file (default: 0)

    Returns:
        List of features in native Python types if no `on_feature` callback is provided; otherwise, an empty list

    Raises:
        ValueError: If the audio data is too short or invalid
        RuntimeError: If processing fails critically
        ExtractionPipelineError: If there is a critical error in the extraction pipeline
        AudioValidationError: If the audio data is invalid
        AudioProcessingError: If an unexpected processing error occurs
    """
    error_stats = ErrorStats()
    error_limiter = ErrorRateLimiter()

    try:
        # Input validation with more specific errors
        if not isinstance(audio_data, np.ndarray):
            raise AudioValidationError(
                message="Invalid audio data type",
                parameter="audio_data",
                expected="numpy.ndarray",
                actual=type(audio_data).__name__,
            )
        if not np.isfinite(audio_data).all():
            raise AudioValidationError("Audio data contains infinite or NaN values")
        if sample_rate <= 0:
            raise AudioValidationError(f"Invalid sample rate: {sample_rate}")
        if channels <= 0:
            raise AudioValidationError(f"Invalid number of channels: {channels}")
        if len(audio_data) < FRAME_LENGTH:
            raise AudioValidationError(
                f"Audio data too short: {len(audio_data)} samples < {FRAME_LENGTH} required"
            )

        # Early return checks
        if feature_config is not None and not any(feature_config.values()):
            logger.warning("No features enabled in configuration")
            return []

        # Convert to float32 for memory efficiency if not already
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Reshape interleaved multi-channel audio if needed
        if audio_data.ndim == 1 and channels > 1:
            samples_per_channel = audio_data.shape[0] // channels
            audio_data = audio_data[: samples_per_channel * channels].reshape(
                (samples_per_channel, channels)
            )

        # Correctly compute duration (samples per channel / sample_rate)
        duration = audio_data.shape[0] / sample_rate
        logger.info(f"Audio length: {audio_data.shape[0]} samples")
        logger.info(f"Audio duration: {duration:.2f} seconds")
        if start_sample > 0:
            logger.info(f"Starting from sample {start_sample} ({start_sample/sample_rate:.3f}s)")

        # Calculate expected frames based on hop length
        n_frames = 1 + (audio_data.shape[0] - FRAME_LENGTH) // HOP_LENGTH
        logger.info(f"Expected number of frames: {n_frames}")

        # Log which features will be computed
        if feature_config is not None:
            enabled_features = [k for k, v in feature_config.items() if v]
            logger.info(f"Computing selected features: {', '.join(enabled_features)}")
        else:
            logger.info("Computing all available features")

        # Create the process function once (constant across batches)
        process_func = partial(
            process_frame,
            sample_rate=sample_rate,
            frame_length=FRAME_LENGTH,
            feature_config=feature_config,
            start_sample=start_sample,  # Pass start_sample to process_frame
        )

        # Optimal resource allocation
        MAX_WORKERS = min(
            calculate_max_workers(audio_data.shape[0], FRAME_LENGTH, HOP_LENGTH),
            mp.cpu_count(),
        )
        # Optimize chunk size calculation
        CHUNK_SIZE = max(
            1, min(BATCH_SIZE // MAX_WORKERS, n_frames // (MAX_WORKERS * 2))
        )

        processed_frames = 0
        valid_features: List[Dict] = [] if on_feature is None else []
        error_count = 0
        MAX_ERRORS = n_frames // 2.5  # Allow up to 2.5% error rate

        try:
            # Process batches with a streaming generator using minimal memory footprint
            with error_tracking_context(error_stats):
                with mp.Pool(processes=MAX_WORKERS) as pool:
                    frames_iter = create_frame_generator(
                        audio_data, audio_data.shape[0], n_frames
                    )

                    # Process batches until no frames remain
                    for batch_frames in iter(
                        lambda: list(islice(frames_iter, BATCH_SIZE)), []
                    ):
                        if error_count > MAX_ERRORS:
                            raise ExtractionPipelineError(
                                error_count=error_count,
                                total_frames=n_frames,
                                error_rate=f"{(error_count/n_frames)*100:.2f}%",
                            )

                        try:
                            # Use imap with optimized chunk size for improved memory usage
                            for _, feature in pool.imap(
                                process_func, batch_frames, chunksize=CHUNK_SIZE
                            ):
                                if feature is not None:
                                    if on_feature is not None:
                                        on_feature(feature)
                                    else:
                                        # Convert to dict excluding None values
                                        valid_features.append(feature.to_dict())
                                else:
                                    if error_limiter.should_log():
                                        logger.warning(
                                            f"Frame processing failed at {processed_frames}"
                                        )
                                    error_count += 1
                                processed_frames += 1

                                # Periodic progress updates
                                if processed_frames % 1000 == 0:
                                    logger.info(
                                        f"Processed {processed_frames}/{n_frames} frames"
                                    )

                        except mp.TimeoutError as e:
                            logger.error(
                                f"Batch processing timeout at frame {processed_frames}"
                            )
                            error_count += 1
                            if error_count > MAX_ERRORS:
                                raise ExtractionPipelineError(
                                    "Too many timeout errors"
                                ) from e
                        except mp.ProcessError as e:
                            logger.error(f"Process error in batch: {str(e)}")
                            error_count += len(batch_frames)
                        except (ValueError, TypeError, RuntimeError) as e:
                            logger.exception(f"Batch processing error: {str(e)}")
                            error_count += len(batch_frames)

                        # Release batch_frames immediately after processing for better memory reclamation.
                        del batch_frames
                        # Optionally, force GC if under heavy load:
                        gc.collect()

                    # Force garbage collection periodically
                    if processed_frames % (BATCH_SIZE * 10) == 0:
                        gc.collect()

        except ExtractionPipelineError:
            raise
        except Exception as e:
            logger.error(f"Pipeline error stats: {error_stats.get_summary()}")
            raise ExtractionPipelineError(
                message="Pipeline execution failed",
                error_count=error_count,
                total_frames=n_frames,
                original_error=str(e),
            ) from e

    except AudioValidationError:
        raise
    except ExtractionPipelineError:
        raise
    except Exception as e:
        raise AudioProcessingError(
            message="Unexpected processing error", details={"error": str(e)}
        ) from e
    finally:
        if error_stats.total_errors > 0:
            logger.info(f"Error statistics:\n{error_stats.get_summary()}")
        gc.collect()

    completion_ratio = processed_frames / n_frames if n_frames > 0 else 0
    if completion_ratio < 0.97:  # Allow for up to 3% frame loss
        logger.warning(
            f"Only processed {completion_ratio * 100:.1f}% of expected frames"
        )

    return valid_features if not on_feature else []
