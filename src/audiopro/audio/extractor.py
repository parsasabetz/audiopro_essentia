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

    # Frame generator for memory efficiency
    def frame_generator():
        for i in range(0, len(audio_data) - FRAME_LENGTH + 1, HOP_LENGTH):
            yield (i, audio_data[i : i + FRAME_LENGTH])

    frames = frame_generator()

    # Precompute common arrays once
    window_func = np.hanning(FRAME_LENGTH).astype(np.float32)
    freq_array = np.fft.rfftfreq(FRAME_LENGTH, d=1 / sample_rate).astype(np.float32)

    # Calculate the maximum number of workers based on the audio data length
    MAX_WORKERS = calculate_max_workers(len(audio_data), FRAME_LENGTH, HOP_LENGTH)
    valid_features = []

    # Process audio data in batches
    total_batches = (len(audio_data) - FRAME_LENGTH + 1) // (
        BATCH_SIZE * HOP_LENGTH
    ) + 1

    # Process each batch of frames in parallel
    with mp.Pool(processes=MAX_WORKERS) as pool:
        for batch_idx in range(total_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(audio_data) - FRAME_LENGTH + 1)
            batch_frames = list(islice(frames, start_idx, end_idx))
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
            except (mp.TimeoutError, mp.ProcessError) as e:
                logger.error("Error processing batch %d: %s", batch_idx + 1, str(e))
                continue

            for _, feature in sorted(batch_results, key=lambda x: x[0]):
                if feature is not None:
                    valid_features.append(feature)

            logger.info("Processed batch %d/%d", batch_idx + 1, total_batches)

    if not valid_features:
        raise ValueError("No valid features could be extracted")

    return optimized_convert_to_native_types(valid_features)
