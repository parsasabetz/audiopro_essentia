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


def extract_features(audio_data: np.ndarray, sample_rate: int) -> list:
    """
    Extracts features from the given audio data.

    This function processes the audio data in frames and extracts features using
    parallel processing for efficiency. It handles audio data that is long enough
    for analysis and processes it in batches to optimize memory usage.

    Args:
        audio_data (np.ndarray): The audio data as a NumPy array.
        sample_rate (int): The sample rate of the audio data.

    Returns:
        list: A list of extracted features.

    Raises:
        ValueError: If the audio data is too short for analysis or if no valid
                    features could be extracted.

    Notes:
        - The function uses a frame generator to process the audio data in frames.
        - It precomputes common arrays like the window function and frequency array
          for efficiency.
        - The maximum number of workers for parallel processing is calculated based
          on the audio data length.
        - The audio data is processed in batches, and each batch is processed in
          parallel using a multiprocessing pool.
        - If an error occurs during batch processing, it logs the error and continues
          with the next batch.
        - The function returns a list of valid features extracted from the audio data.
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
    total_batches = (total_frames + BATCH_SIZE - 1) // BATCH_SIZE
    valid_features = []
    processed_frames = 0

    # Process batches with progress tracking
    with mp.Pool(processes=MAX_WORKERS) as pool:
        frames = frame_generator()
        for batch_idx in range(total_batches):
            batch_frames = list(islice(frames, BATCH_SIZE))
            if not batch_frames:
                break

            process_func = partial(
                process_frame,
                sample_rate=sample_rate,
                frame_length=FRAME_LENGTH,
                window_func=window_func,
                freq_array=freq_array,
            )

            try:
                batch_results = pool.map(process_func, batch_frames)
                # Track valid results
                for _, feature in sorted(batch_results, key=lambda x: x[0]):
                    if feature is not None:
                        processed_frames += 1
                        valid_features.append(feature)
            except (mp.TimeoutError, mp.ProcessError) as e:
                logger.error(f"Error processing batch {batch_idx + 1}: {str(e)}")
                continue

            logger.info(
                f"Processed batch {batch_idx + 1}/{total_batches} ({processed_frames}/{total_frames} frames)"
            )

    # Validate processing completion
    if not valid_features:
        raise ValueError("No valid features could be extracted")

    completion_ratio = processed_frames / total_frames
    if completion_ratio < 0.95:  # Allow for up to 5% frame loss
        logger.warning(f"Only processed {completion_ratio:.1%} of expected frames")

    logger.info(f"Successfully processed {processed_frames} frames")
    return optimized_convert_to_native_types(valid_features)
