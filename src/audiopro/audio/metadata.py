"""
Module for handling audio file metadata extraction.
"""

# Typing imports
from typing import Dict

# Standard library imports
import os
import datetime
from pathlib import Path
import hashlib
import mimetypes

# Third-party imports
import numpy as np

# Local imports
from audiopro.utils.logger import get_logger

# Configure logging
logger = get_logger()


def calculate_file_hash(file_path: str, block_size=1048576) -> str:
    """
    Calculate the SHA-256 hash of a file synchronously.

    Args:
        file_path (str): The path to the file for which the hash is to be calculated.
        block_size (int, optional): The size of each block to read from the file. Defaults to 1048576 bytes (1 MB).

    Returns:
        str: The SHA-256 hash of the file in hexadecimal format. If an error occurs, returns "hash_calculation_failed".

    Raises:
        IOError: If an I/O error occurs while reading the file.
        OSError: If an OS-related error occurs while reading the file.
    """

    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(block_size), b""):
                sha256_hash.update(block)
        return sha256_hash.hexdigest()
    except (IOError, OSError) as e:
        logger.error("Failed to calculate file hash: %s", e)
        return "hash_calculation_failed"


def get_file_metadata(file_path: str, audio_data: np.ndarray, sample_rate: int) -> Dict:
    """
    Extracts and returns metadata information from an audio file.

    Args:
        file_path (str): The path to the audio file.
        audio_data (np.ndarray): The audio data as a NumPy array.
        sample_rate (int): The sample rate of the audio data.

    Returns:
        Dict: A dictionary containing file and audio metadata, including:
            - file_info:
                - filename (str): The name of the file.
                - format (str): The file format (extension).
                - size_mb (float): The size of the file in megabytes.
                - created_date (str): The creation date of the file in ISO format.
                - mime_type (str): The MIME type of the file.
                - sha256_hash (str): The SHA-256 hash of the file.
            - audio_info:
                - duration_seconds (float): The duration of the audio in seconds.
                - sample_rate (int): The sample rate of the audio.
                - channels (int): The number of audio channels.
                - peak_amplitude (float): The peak amplitude of the audio.
                - rms_amplitude (float): The root mean square (RMS) amplitude of the audio.
                - dynamic_range_db (float): The dynamic range of the audio in decibels.
                - quality_metrics:
                    - dc_offset (float): The DC offset of the audio.
                    - silence_ratio (float): The ratio of silent samples in the audio.
                    - potentially_clipped_samples (int): The number of potentially clipped samples.
    """

    file_stats = os.stat(file_path)
    path_obj = Path(file_path)

    # Single pass calculations
    abs_audio = np.abs(audio_data)
    peak_amplitude = float(np.max(abs_audio))
    rms_value = float(np.sqrt(np.mean(audio_data**2)))

    return {
        "file_info": {
            "filename": path_obj.name,
            "format": path_obj.suffix[1:],
            "size_mb": file_stats.st_size / (1024**2),
            "created_date": datetime.datetime.fromtimestamp(
                file_stats.st_ctime
            ).isoformat(),
            "mime_type": mimetypes.guess_type(file_path)[0] or "unknown",
            "sha256_hash": calculate_file_hash(file_path),
        },
        "audio_info": {
            "duration_seconds": len(audio_data) / sample_rate,
            "sample_rate": sample_rate,
            "channels": 1 if audio_data.ndim == 1 else audio_data.shape[1],
            "peak_amplitude": peak_amplitude,
            "rms_amplitude": rms_value,
            "dynamic_range_db": 20
            * np.log10(
                (peak_amplitude + np.finfo(float).eps)
                / (rms_value + np.finfo(float).eps)
            ),
            "quality_metrics": {
                "dc_offset": float(np.mean(audio_data)),
                "silence_ratio": float(np.mean(abs_audio < 0.001)),
                "potentially_clipped_samples": int(np.sum(abs_audio > 0.99)),
            },
        },
    }
