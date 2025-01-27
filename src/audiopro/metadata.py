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

def calculate_file_hash(file_path: str, block_size=65536) -> str:
    """Calculate SHA-256 hash of file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            sha256_hash.update(block)
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
    
    # Calculate duration and audio metrics
    duration = len(audio_data) / sample_rate
    peak_amplitude = np.max(np.abs(audio_data))
    rms_value = np.sqrt(np.mean(audio_data**2))
    
    # Calculate dynamic range
    dynamic_range_db = 20 * np.log10(peak_amplitude / (rms_value + 1e-9)) if peak_amplitude > 0 else 0

    metadata = {
        "file_info": {
            "filename": file_path_obj.name,
            "format": file_path_obj.suffix[1:],
            "size_mb": round(file_stats.st_size / (1024 * 1024), 2),
            "created_date": datetime.datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            "modified_date": datetime.datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            "mime_type": mimetypes.guess_type(file_path)[0],
            "sha256_hash": calculate_file_hash(file_path)
        },
        "audio_info": {
            "duration_seconds": float(duration),
            "duration_formatted": str(datetime.timedelta(seconds=int(duration))),
            "channels": 1 if len(audio_data.shape) == 1 else audio_data.shape[1],
            "peak_amplitude": float(peak_amplitude),
            "dynamic_range_db": float(dynamic_range_db),
            "quality_metrics": {
                "silence_percentage": float(np.mean(np.abs(audio_data) < 0.0001) * 100),
                "potentially_clipped": bool(np.any(np.abs(audio_data) > 0.99)),
                "sample_rate_category": "high" if sample_rate >= 44100 else "low"
            }
        }
    }
    
    return metadata