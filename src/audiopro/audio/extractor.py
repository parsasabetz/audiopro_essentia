"""
Module for audio feature extraction.
Features are extracted per frame with multiprocessing, preserving output order.
"""

# Standard library imports
import logging
from functools import partial
import multiprocessing as mp

# Third-party imports
import numpy as np
import essentia.standard as es

# Typing imports
from typing import Dict, List, Optional, Tuple

# Local util imports
from ..utils import (
    compute_spectral_bandwidth,
    optimized_convert_to_native_types,
)

# Set up logging
logger = logging.getLogger(__name__)

# Constants for audio processing
FRAME_LENGTH = 2048
HOP_LENGTH = 512
BATCH_SIZE = 1000  # Process frames in batches
FREQUENCY_BANDS = {
    "sub_bass": (20, 60),
    "bass": (60, 250),
    "low_mid": (250, 500),
    "mid": (500, 2000),
    "upper_mid": (2000, 5000),
    "treble": (5000, 20000),
}


# Calculate max workers dynamically based on workload
def calculate_max_workers(
    audio_data_length: int, frame_length: int, hop_length: int
) -> int:
    num_frames = (audio_data_length - frame_length) // hop_length + 1
    return min(32, max(1, num_frames // 1000))


# Cache frequency bins to avoid recalculating for each frame
FREQS_CACHE = {}


def compute_frequency_bands(
    spec: np.ndarray, sample_rate: int, frame_length: int
) -> Dict[str, float]:
    """
    Compute frequency bands with proper frequency bin calculation.

    Args:
        spec: Magnitude spectrum
        sample_rate: Sample rate of the audio
        frame_length: Length of the frame

    Returns:
        Dictionary containing the energy in each frequency band
    """
    key = (sample_rate, frame_length)
    if key not in FREQS_CACHE:
        FREQS_CACHE[key] = np.linspace(0, sample_rate / 2, len(spec))
    freqs = FREQS_CACHE[key]

    result = {}
    for band_name, (low, high) in FREQUENCY_BANDS.items():
        # Create mask for the current frequency band
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            # Calculate mean energy in the band
            result[band_name] = float(np.mean(spec[mask]))
        else:
            # Handle case where no frequencies fall in the band
            result[band_name] = 0.0

    return result


def process_frame(
    frame_data: Tuple[int, np.ndarray],
    sample_rate: int,
    frame_length: int,
    window_func: np.ndarray,
    freq_array: np.ndarray,
) -> Tuple[int, Optional[Dict]]:
    """Process a single audio frame with better error handling"""
    frame_index, frame = frame_data

    # Validate frame data
    if frame.size == 0 or np.all(np.isnan(frame)):
        logger.warning(f"Invalid frame data at index {frame_index}")
        return frame_index, None

    try:
        # Ensure FRAME_LENGTH is even
        assert frame_length % 2 == 0, "FRAME_LENGTH must be even."

        # Window and pad in one step to avoid memory duplication
        if len(frame) < frame_length:
            if len(frame) < frame_length // 2:  # Skip if too short
                return frame_index, None
            frame = np.pad(frame, (0, frame_length - len(frame))).astype(
                np.float32
            )  # Ensure float32
            logger.debug(f"Padded frame {frame_index} to length {frame_length}.")

        # Additional check to ensure frame length is even
        if len(frame) % 2 != 0:
            frame = np.append(frame, 0.0).astype(np.float32)  # Ensure float32
            logger.debug(f"Appended zero to frame {frame_index} to make length even.")

        frame *= window_func  # Use precomputed window function

        # Replace window + STFT with Essentia
        w = es.Windowing(type="hann")
        spectrum = es.Spectrum()
        windowed_frame = w(frame)
        spec = spectrum(windowed_frame)

        if np.all(spec == 0):
            return frame_index, None

        # Use Essentia algorithms for feature extraction
        centroid = es.Centroid(range=sample_rate / 2)(spec)

        # Manually compute spectral bandwidth
        spectral_bandwidth = compute_spectral_bandwidth(spec, freq_array, centroid)

        flatness = es.Flatness()(spec)  # Updated to use spectrum

        # Replace RollOff without 'cutFrequency'
        rolloff = es.RollOff()(spec)  # Removed cutFrequency parameter

        zcr = es.ZeroCrossingRate()(frame)

        mfcc_alg = es.MFCC(numberCoefficients=13)
        _, mfcc_out = mfcc_alg(spectrum(windowed_frame))

        # Initialize algorithms for chroma features
        window = es.Windowing(type="blackmanharris92")
        spectrum_complex = es.Spectrum(size=frame_length)
        spectral_peaks = es.SpectralPeaks(
            orderBy="magnitude",
            magnitudeThreshold=0.00001,
            minFrequency=20,
            maxFrequency=sample_rate / 2,
            maxPeaks=100,
        )
        hpcp = es.HPCP(
            size=12,
            referenceFrequency=440,
            bandPreset=False,
            minFrequency=20,
            maxFrequency=sample_rate / 2,
            weightType="squaredCosine",  # Fixed: changed from 'squared cosine' to 'squaredCosine'
            nonLinear=False,
            normalized="unitMax",
        )

        # Reuse windowed frame for spectrum_complex to avoid reapplying the window
        windowed = window(frame)
        frame_spectrum = spectrum_complex(windowed)
        freqs, mags = spectral_peaks(frame_spectrum)

        # Compute HPCP (chroma) features with error handling
        try:
            if len(freqs) > 0:  # Only compute if peaks were found
                chroma_vector = hpcp(freqs, mags)
            else:
                chroma_vector = np.zeros(12, dtype=np.float32)

            # Normalize chroma vector
            chroma_sum = np.sum(chroma_vector)
            if chroma_sum > 0:
                chroma_vector = chroma_vector / chroma_sum
        except Exception as e:
            logger.debug(f"HPCP calculation failed for frame {frame_index}: {e}")
            chroma_vector = np.zeros(12, dtype=np.float32)

        try:
            # Calculate frequency bands using the cached frequency bins
            freq_bands = compute_frequency_bands(spec, sample_rate, frame_length)

            feature_dict = {
                "time": float(frame_index * HOP_LENGTH / sample_rate),
                "rms": float(np.sqrt(np.mean(frame**2))),
                "spectral_centroid": float(centroid),
                "spectral_bandwidth": float(spectral_bandwidth),
                "spectral_flatness": float(flatness),
                "spectral_rolloff": float(rolloff),
                "zero_crossing_rate": float(zcr),
                "mfcc": [float(x) for x in mfcc_out],
                "frequency_bands": freq_bands,
                "chroma": [float(x) for x in chroma_vector],
            }
            return frame_index, feature_dict

        except Exception as e:
            logger.error(f"Frame processing error at {frame_index}: {str(e)}")
            return frame_index, None

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

    # Precompute window_func and freq_array once:
    window_func = np.hanning(FRAME_LENGTH).astype(np.float32)
    freq_array = np.fft.rfftfreq(FRAME_LENGTH, d=1 / sample_rate).astype(np.float32)

    # Calculate max workers dynamically
    MAX_WORKERS = calculate_max_workers(len(audio_data), FRAME_LENGTH, HOP_LENGTH)

    # Process frames in batches
    valid_features = []
    total_batches = (len(frames) + BATCH_SIZE - 1) // BATCH_SIZE

    with mp.Pool(processes=MAX_WORKERS) as pool:
        for batch_idx in range(total_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(frames))
            batch_frames = frames[start_idx:end_idx]

            try:
                process_func = partial(
                    process_frame,
                    sample_rate=sample_rate,
                    frame_length=FRAME_LENGTH,
                    window_func=window_func,
                    freq_array=freq_array,
                )

                batch_results = pool.map(process_func, batch_frames)

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
    return optimized_convert_to_native_types(valid_features)
