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


def calculate_file_hash(file_path: str, block_size=1048576) -> str:
    """Calculate SHA-256 hash of file synchronously"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            sha256_hash.update(block)
    return sha256_hash.hexdigest()


def get_file_metadata(file_path: str, audio_data: np.ndarray, sample_rate: int) -> Dict:
    """Simplified metadata extraction"""
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
            "created_date": datetime.datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            "mime_type": mimetypes.guess_type(file_path)[0] or "unknown",
            "sha256_hash": calculate_file_hash(file_path),
        },
        "audio_info": {
            "duration_seconds": len(audio_data) / sample_rate,
            "sample_rate": sample_rate,
            "channels": 1 if audio_data.ndim == 1 else audio_data.shape[1],
            "peak_amplitude": peak_amplitude,
            "rms_amplitude": rms_value,
            "dynamic_range_db": 20 * np.log10((peak_amplitude + np.finfo(float).eps) / 
                                            (rms_value + np.finfo(float).eps)),
            "quality_metrics": {
                "dc_offset": float(np.mean(audio_data)),
                "silence_ratio": float(np.mean(abs_audio < 0.001)),
                "potentially_clipped_samples": int(np.sum(abs_audio > 0.99))
            },
        },
    }
