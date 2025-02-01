"""
Module for audio feature extraction.
Features are extracted per frame with multiprocessing, preserving output order.
"""

# Standard library imports
from functools import partial
import multiprocessing as mp
from itertools import islice

# Third-party imports
import numpy as np

# Local util imports
from audiopro.utils.logger import get_logger
from audiopro.utils import (
    optimized_convert_to_native_types,
    calculate_max_workers,
    FRAME_LENGTH,
    HOP_LENGTH,
    BATCH_SIZE,
)
from .feature_utils import process_frame

# Set up logger
logger = get_logger()


def extract_features(audio_data: np.ndarray, sample_rate: int, on_feature=None) -> list:
    """
    Extracts features from the given audio data by processing it in frames.

    The audio data is divided into overlapping frames according to FRAME_LENGTH and HOP_LENGTH.
    Each frame is preprocessed with a window function and transformed using FFT to extract spectral features.
    Multiprocessing is used to distribute frame processing across multiple processes, ensuring a low CPU and memory footprint.
    If an on_feature callback is provided, valid features are dispatched immediately via the callback,
    avoiding the accumulation of large in-memory lists.

    Args:
        audio_data (np.ndarray): The audio data as a numpy array.
        sample_rate (int): The sample rate of the audio.
        on_feature (callable, optional): A function to receive each extracted feature immediately.
            When provided, the function returns an empty list, and the caller is responsible for handling features.

    Returns:
        list: A list of features in native Python types if no on_feature callback is provided; otherwise, an empty list.

    Raises:
        ValueError: If the audio data is too short for analysis or if no valid features can be extracted.

    Notes:
        - Uses a streaming frame generator and batches processing to minimize memory usage.
        - Precomputes and reuses window and frequency arrays for efficiency.
        - Logs progress and warns if processed frames fall below an acceptable completion ratio.
    """

    if len(audio_data) < FRAME_LENGTH:
        raise ValueError("Audio data too short for analysis")

    # Precise frame calculation
    total_samples = len(audio_data)
    max_start_idx = total_samples - FRAME_LENGTH
    total_frames = (max_start_idx + HOP_LENGTH) // HOP_LENGTH
    expected_duration = total_samples / sample_rate

    logger.info(f"Audio length: {total_samples} samples")
    logger.info(f"Audio duration: {expected_duration:.2f} seconds")
    logger.info(f"Expected number of frames: {total_frames}")

    # Improved frame generator with boundary checking
    def frame_generator():
        for frame_idx in range(total_frames):
            start_idx = frame_idx * HOP_LENGTH
            if start_idx + FRAME_LENGTH > total_samples:
                break
            frame_data = audio_data[start_idx : start_idx + FRAME_LENGTH]
            if len(frame_data) == FRAME_LENGTH:  # Ensure complete frames
                yield (frame_idx, frame_data)

    # Precompute arrays once
    window_func = np.hanning(FRAME_LENGTH).astype(np.float32)
    freq_array = np.fft.rfftfreq(FRAME_LENGTH, d=1 / sample_rate).astype(np.float32)

    # Optimal batch size calculation
    MAX_WORKERS = calculate_max_workers(total_samples, FRAME_LENGTH, HOP_LENGTH)
    processed_frames = 0
    valid_features = []

    # Process batches with a streaming generator using minimal memory footprint
    with mp.Pool(processes=MAX_WORKERS) as pool:
        frames_iter = frame_generator()
        # Process batches until no frames remain
        for batch_frames in iter(lambda: list(islice(frames_iter, BATCH_SIZE)), []):
            process_func = partial(
                process_frame,
                sample_rate=sample_rate,
                frame_length=FRAME_LENGTH,
                window_func=window_func,
                freq_array=freq_array,
            )
            try:
                # Use imap with a chunksize for improved memory usage
                for _, feature in pool.imap(process_func, batch_frames, chunksize=100):
                    if feature is not None:
                        processed_frames += 1
                        if on_feature is not None:
                            # Immediately dispatch the feature instead of storing
                            on_feature(feature)
                        else:
                            valid_features.append(feature)
                    else:
                        processed_frames += 1
            except (mp.TimeoutError, mp.ProcessError) as e:
                logger.error("Error processing a batch: %s", str(e))
                continue

            logger.info("Processed %d/%d frames", processed_frames, total_frames)

    if on_feature is None and not valid_features:
        raise ValueError("No valid features could be extracted")

    completion_ratio = processed_frames / total_frames
    if completion_ratio < 0.97:  # Allow for up to 3% frame loss
        logger.warning(
            "Only processed %.1f%% of expected frames", completion_ratio * 100
        )

    logger.info("Successfully processed %d frames", processed_frames)
    if on_feature is not None:
        return []
    return optimized_convert_to_native_types(valid_features)
