"""
Module for audio feature extraction.
Features are extracted per frame with parallel processing, preserving output order.
"""

# Standard library imports
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os

# Third-party imports
import numpy as np
import librosa

# Typing imports
from typing import Dict, List, Optional, Tuple

# Set up logging
logger = logging.getLogger(__name__)

# Constants for audio processing
FRAME_LENGTH = 2048
HOP_LENGTH = 512
FREQUENCY_BANDS = {
    "sub_bass": (20, 60),
    "bass": (60, 250),
    "low_mid": (250, 500),
    "mid": (500, 2000),
    "upper_mid": (2000, 5000),
    "treble": (5000, 20000),
}

# Add max workers constant to prevent resource exhaustion
MAX_WORKERS = min(32, (os.cpu_count() or 1) + 4)
BATCH_SIZE = 1000  # Process frames in batches


def convert_to_native_types(data):
    """Convert numpy types to native Python types"""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)
    elif isinstance(data, dict):
        return {key: convert_to_native_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_native_types(item) for item in data]
    return data


def process_frame(
    frame_data: Tuple[int, np.ndarray], sample_rate: int, frame_length: int
) -> Tuple[int, Optional[Dict]]:
    """Process a single audio frame with better error handling"""
    frame_index, frame = frame_data

    # Validate frame data
    if frame.size == 0 or np.all(np.isnan(frame)):
        logger.warning(f"Invalid frame data at index {frame_index}")
        return frame_index, None

    try:
        # Window and pad in one step to avoid memory duplication
        if len(frame) < frame_length:
            if len(frame) < frame_length // 2:  # Skip if too short
                return frame_index, None
            frame = np.pad(frame, (0, frame_length - len(frame)))
        frame *= np.hanning(len(frame))

        # Compute STFT with error check
        spec = np.abs(librosa.stft(frame, n_fft=frame_length, window="hann"))
        if not np.any(spec) or np.any(np.isnan(spec)):
            return frame_index, None

        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=frame_length)

        feature_dict = {
            "time": float(
                librosa.frames_to_time(
                    frame_index // HOP_LENGTH, sr=sample_rate, hop_length=HOP_LENGTH
                )
            ),
            "rms": float(np.sqrt(np.mean(frame**2))),
            "spectral_centroid": float(
                librosa.feature.spectral_centroid(S=spec, sr=sample_rate).mean()
            ),
            "spectral_bandwidth": float(
                librosa.feature.spectral_bandwidth(S=spec, sr=sample_rate).mean()
            ),
            "spectral_flatness": float(
                librosa.feature.spectral_flatness(y=frame).mean()
            ),
            "spectral_rolloff": float(
                librosa.feature.spectral_rolloff(S=spec, sr=sample_rate).mean()
            ),
            "zero_crossing_rate": float(
                librosa.feature.zero_crossing_rate(frame).mean()
            ),
            "mfcc": [
                float(x)
                for x in librosa.feature.mfcc(y=frame, sr=sample_rate, n_mfcc=13).mean(
                    axis=1
                )
            ],
            "frequency_bands": {
                band_name: float(
                    np.mean(spec[(freqs >= freq_range[0]) & (freqs < freq_range[1])])
                )
                for band_name, freq_range in FREQUENCY_BANDS.items()
            },
        }

        # Safe chroma calculation
        chroma = np.zeros(12)  # Initialize with zeros
        if np.any(spec > 0):  # Only compute if we have signal
            try:
                chroma = librosa.feature.chroma_stft(
                    S=spec,
                    sr=sample_rate,
                    tuning=0.0,
                    norm=None,  # Avoid normalization issues
                    n_chroma=12,
                    n_fft=frame_length,
                ).mean(axis=1)
            except Exception as e:
                logger.debug(f"Chroma calculation failed: {e}")

        feature_dict["chroma"] = chroma.tolist()
        return frame_index, feature_dict

    except Exception as e:
        logger.error(f"Frame processing error at {frame_index}: {str(e)}")
        return frame_index, None


def extract_features(audio_data: np.ndarray, sample_rate: int) -> List[Dict]:
    """Extract features with batched processing to prevent timeouts"""
    if len(audio_data) < FRAME_LENGTH:
        raise ValueError("Audio data too short for analysis")

    # Calculate frames
    try:
        frames = [
            (i, audio_data[i : i + FRAME_LENGTH])
            for i in range(0, len(audio_data) - FRAME_LENGTH + 1, HOP_LENGTH)
        ]
    except Exception as e:
        raise ValueError(f"Failed to create frames: {str(e)}")

    if not frames:
        raise ValueError("No frames could be extracted from audio data")

    # Process frames in batches
    valid_features = []
    total_batches = (len(frames) + BATCH_SIZE - 1) // BATCH_SIZE

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for batch_idx in range(total_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(frames))
            batch_frames = frames[start_idx:end_idx]

            try:
                process_func = partial(
                    process_frame, sample_rate=sample_rate, frame_length=FRAME_LENGTH
                )

                batch_results = list(executor.map(process_func, batch_frames))

                # Process valid results from this batch
                for idx, feature in sorted(batch_results, key=lambda x: x[0]):
                    if feature is not None and isinstance(feature, dict):
                        valid_features.append(feature)

                logger.info(f"Processed batch {batch_idx + 1}/{total_batches}")

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx + 1}: {str(e)}")
                continue  # Continue with next batch instead of failing completely

    if not valid_features:
        raise ValueError("No valid features could be extracted")

    # Convert all numpy types to native Python types before returning
    return convert_to_native_types(valid_features)
