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
        if start_idx + FRAME_LENGTH > total_samples:
            break
        frame_data = audio_data[start_idx : start_idx + FRAME_LENGTH].copy()
        if len(frame_data) == FRAME_LENGTH:
            yield (frame_idx, frame_data)
        # Explicitly delete frame data to help garbage collection
        del frame_data


def extract_features(
    audio_data: NDArray[np.float32],
    sample_rate: int,
    on_feature: Optional[Callable[[FrameFeatures], None]] = None,
    feature_config: Optional[FeatureConfig] = None,
) -> List[Dict]:
    """
    Extracts features from the given audio data by processing it in frames.

    The audio data is divided into overlapping frames according to FRAME_LENGTH and HOP_LENGTH.
    Each frame is preprocessed with a window function and transformed using FFT to extract spectral features.
    Multiprocessing is used to distribute frame processing across multiple processes, ensuring a low CPU and memory footprint.

    Args:
        audio_data: The audio data as a numpy array
        sample_rate: The sample rate of the audio
        on_feature: Optional callback for immediate feature processing
        feature_config: Optional configuration specifying which features to compute.
                      If None, all features will be computed.

    Returns:
        List of features in native Python types if no on_feature callback is provided; otherwise, an empty list

    Raises:
        ValueError: If the audio data is too short or invalid
        RuntimeError: If processing fails critically
    """
    if not isinstance(audio_data, np.ndarray):
        raise ValueError("Audio data must be a numpy array")

    if len(audio_data) < FRAME_LENGTH:
        raise ValueError(
            f"Audio data too short for analysis: {len(audio_data)} samples < {FRAME_LENGTH} required"
        )

    # Convert to float32 for memory efficiency if not already
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # Precise frame calculation
    total_samples = len(audio_data)
    max_start_idx = total_samples - FRAME_LENGTH
    total_frames = (max_start_idx + HOP_LENGTH) // HOP_LENGTH
    expected_duration = total_samples / sample_rate

    logger.info(f"Audio length: {total_samples} samples")
    logger.info(f"Audio duration: {expected_duration:.2f} seconds")
    logger.info(f"Expected number of frames: {total_frames}")

    # Log which features will be computed
    if feature_config is not None:
        enabled_features = [k for k, v in feature_config.items() if v]
        logger.info(f"Computing selected features: {', '.join(enabled_features)}")
    else:
        logger.info("Computing all available features")

    # Precompute arrays once and ensure they're float32
    window_func = np.hanning(FRAME_LENGTH).astype(np.float32)
    freq_array = np.fft.rfftfreq(FRAME_LENGTH, d=1 / sample_rate).astype(np.float32)

    # Create the process function once (constant across batches)
    process_func = partial(
        process_frame,
        sample_rate=sample_rate,
        frame_length=FRAME_LENGTH,
        window_func=window_func,
        freq_array=freq_array,
        feature_config=feature_config,
    )

    # Optimal resource allocation
    MAX_WORKERS = min(
        calculate_max_workers(total_samples, FRAME_LENGTH, HOP_LENGTH), mp.cpu_count()
    )
    CHUNK_SIZE = max(
        1, min(100, total_frames // (MAX_WORKERS * 4))
    )  # Adaptive chunk size

    processed_frames = 0
    valid_features: List[Dict] = [] if on_feature is None else []
    error_count = 0
    MAX_ERRORS = total_frames // 2.5  # Allow up to 2.5% error rate

    try:
        # Process batches with a streaming generator using minimal memory footprint
        with mp.Pool(processes=MAX_WORKERS) as pool:
            frames_iter = create_frame_generator(
                audio_data, total_samples, total_frames
            )

            # Process batches until no frames remain
            for batch_frames in iter(lambda: list(islice(frames_iter, BATCH_SIZE)), []):
                if error_count > MAX_ERRORS:
                    raise RuntimeError(f"Too many processing errors: {error_count}")

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
                            error_count += 1
                        processed_frames += 1

                        # Periodic progress updates
                        if processed_frames % 1000 == 0:
                            logger.info(
                                f"Processed {processed_frames}/{total_frames} frames"
                            )

                except (mp.TimeoutError, mp.ProcessError) as e:
                    error_count += len(batch_frames)
                    logger.error(f"Batch processing error: {str(e)}")
                    continue

                # Force garbage collection periodically
                if processed_frames % (BATCH_SIZE * 10) == 0:
                    gc.collect()

    except Exception as e:
        logger.error(f"Critical error during processing: {str(e)}")
        raise RuntimeError(f"Feature extraction failed: {str(e)}") from e

    finally:
        # Clean up
        del window_func, freq_array
        gc.collect()

    completion_ratio = processed_frames / total_frames
    if completion_ratio < 0.97:  # Allow for up to 3% frame loss
        logger.warning(
            f"Only processed {completion_ratio * 100:.1f}% of expected frames"
        )

    if not on_feature:
        if not valid_features:
            raise ValueError("No valid features could be extracted")
        return valid_features  # Already in native types since we used to_dict()
    return []
