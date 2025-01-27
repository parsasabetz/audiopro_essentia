"""
Module for audio feature extraction.
Features are extracted per frame with parallel processing, preserving output order.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import numpy as np
import librosa
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

FRAME_LENGTH = 2048
HOP_LENGTH = 512
FREQUENCY_BANDS = {
    "low": 250,
    "mid": 2000,
}

def process_frame(frame_data: Tuple[int, np.ndarray], sample_rate: int, frame_length: int) -> Tuple[int, Optional[Dict]]:
    """
    Process a single audio frame, extracting features such as RMS, spectral centroid, etc.

    Args:
        frame_data: (frame index, audio frame data).
        sample_rate: Sample rate of the audio.
        frame_length: Expected frame size for STFT.
    Returns:
        (frame index, feature dictionary or None on error).
    """
    frame_index, frame = frame_data

    # Zero-pad frame if needed
    if len(frame) < frame_length:
        frame = np.pad(frame, (0, frame_length - len(frame)))

    try:
        # Spectral calculations
        spec = np.abs(librosa.stft(frame, n_fft=frame_length))
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=frame_length)

        # Feature extraction
        feature_dict = {
            "time": float(librosa.frames_to_time(frame_index // HOP_LENGTH, sr=sample_rate, hop_length=HOP_LENGTH)),
            "rms": float(np.sqrt(np.mean(frame**2))),
            "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(S=spec))),
            "spectral_bandwidth": float(np.mean(librosa.feature.spectral_bandwidth(S=spec))),
            "frequency_bands": {
                "low": [float(np.mean(spec[freqs <= FREQUENCY_BANDS["low"]]))],
                "mid": [float(np.mean(spec[(freqs > FREQUENCY_BANDS["low"]) & 
                                         (freqs <= FREQUENCY_BANDS["mid"])]))],  # Fixed missing bracket here
                "high": [float(np.mean(spec[freqs > FREQUENCY_BANDS["mid"]]))]
            }
        }
        return frame_index, feature_dict

    except Exception as e:
        logger.warning(f"Error processing frame at {frame_index}: {str(e)}")
        return frame_index, None

def extract_features(audio_data: np.ndarray, sample_rate: int) -> List[Dict]:
    """
    Splits the audio into frames, processes them in parallel, and preserves the frame order.

    Args:
        audio_data: Audio time series.
        sample_rate: Sample rate of audio_data.
    Returns:
        List of per-frame feature dictionaries in chronological order.
    """
    # Prepare frames for parallel processing
    frames = [(i, audio_data[i:i + FRAME_LENGTH]) 
             for i in range(0, len(audio_data) - FRAME_LENGTH, HOP_LENGTH)]
    
    # ThreadPoolExecutor spawns parallel tasks for each frame
    with ThreadPoolExecutor() as executor:
        process_func = partial(process_frame, sample_rate=sample_rate, frame_length=FRAME_LENGTH)
        results = list(executor.map(process_func, frames))

    # Sorting results by frame index ensures deterministic output
    # Maintain order and filter None results
    return [feature for _, feature in sorted(results, key=lambda x: x[0]) 
            if feature is not None]