"""
Module for handling audio file metadata extraction.
"""

# Standard library imports
import os
import datetime
from pathlib import Path
import hashlib
import mimetypes

# Third-party imports
import numpy as np

# Typing imports
from typing import Dict


def calculate_file_hash(
    file_path: str, block_size=1048576
) -> str:  # Increased block_size to 1MB
    """Calculate SHA-256 hash of file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(block_size), b""):
                sha256_hash.update(block)
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading file {file_path}: {str(e)}")
    return sha256_hash.hexdigest()


def get_file_metadata(file_path: str, audio_data: np.ndarray, sample_rate: int) -> Dict:
    """
    Extract relevant metadata about an audio file.

    Args:
        file_path: Path to the audio file
        audio_data: Loaded audio data array
        sample_rate: Audio sample rate

    Returns:
        Dictionary containing essential file and audio metadata
    """
    file_stats = os.stat(file_path)
    file_path_obj = Path(file_path)

    # More precise duration calculation
    duration = len(audio_data) / float(sample_rate)

    # Normalize audio data for consistent measurements
    normalized_audio = audio_data / (np.max(np.abs(audio_data)) + np.finfo(float).eps)

    # Accurate peak and RMS calculations using vectorization
    peak_amplitude = float(np.max(np.abs(audio_data)))
    rms_value = float(np.sqrt(np.mean(audio_data**2)))

    # More precise dynamic range calculation with protection against log(0)
    eps = np.finfo(float).eps  # Smallest positive float value
    dynamic_range_db = float(20 * np.log10((peak_amplitude + eps) / (rms_value + eps)))

    metadata = {
        "file_info": {
            "filename": file_path_obj.name,
            "format": file_path_obj.suffix[1:],
            "size_mb": float(
                round(file_stats.st_size / (1024**2), 6)
            ), # for MB
            "created_date": datetime.datetime.fromtimestamp(
                file_stats.st_ctime
            ).isoformat(),
            "modified_date": datetime.datetime.fromtimestamp(
                file_stats.st_mtime
            ).isoformat(),
            "mime_type": mimetypes.guess_type(file_path)[0]
            or "unknown",  # Handle unknown MIME types
            "sha256_hash": calculate_file_hash(file_path),
        },
        "audio_info": {
            "duration_seconds": float(round(duration, 6)),  # More precise duration
            "duration_formatted": str(datetime.timedelta(seconds=int(duration))),
            "sample_rate": int(sample_rate),  # Ensure integer
            "channels": int(1 if audio_data.ndim == 1 else audio_data.shape[1]),
            "peak_amplitude": float(round(peak_amplitude, 6)),
            "rms_amplitude": float(round(rms_value, 6)),
            "dynamic_range_db": float(round(dynamic_range_db, 2)),
            "quality_metrics": {
                "dc_offset": float(round(np.mean(audio_data), 6)),
                "silence_ratio": float(
                    round(np.mean(np.abs(normalized_audio) < 0.001), 4)
                ),
                "potentially_clipped_samples": int(np.sum(np.abs(audio_data) > 0.99)),
                "sample_rate_category": "high" if sample_rate >= 44100 else "low",
            },
        },
    }

    return metadata
