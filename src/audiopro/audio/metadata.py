"""
Module for handling audio file metadata extraction.
"""

# Standard library imports
import datetime
import numpy as np

# Local typing imports
from audiopro.output.types import Metadata
from audiopro.audio.audio_loader import LoaderMetadata

# Local imports
from audiopro.utils.logger import get_logger

# Configure logging
logger = get_logger()


def get_file_metadata(
    audio_data: np.ndarray,
    loader_metadata: LoaderMetadata,  # updated parameter type
) -> Metadata:  # updated return type
    """
    Extracts and returns metadata information from an audio file.

    Args:
        audio_data (np.ndarray): The audio data as a NumPy array.
        loader_metadata (LoaderMetadata, optional): Precomputed metadata from a loader. Defaults to None.

    Returns:
        Metadata: A dictionary containing file and audio metadata, including:
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

    # Single pass calculations for audio metrics
    abs_audio = np.abs(audio_data)
    peak_amplitude = float(np.max(abs_audio))
    rms_value = float(np.sqrt(np.mean(audio_data**2)))
    sample_rate = loader_metadata["sample_rate"]

    return {
        "file_info": {
            "filename": loader_metadata["filename"],
            "format": loader_metadata["format"],
            "codec": loader_metadata["codec"],
            "size_mb": loader_metadata["size_mb"],
            "created_date": datetime.datetime.fromtimestamp(
                loader_metadata["created_date"]
            ).isoformat(),
            "mime_type": loader_metadata["mime_type"],
            "md5_hash": loader_metadata["md5_hash"],
        },
        "audio_info": {
            "duration_seconds": audio_data.shape[0] / (sample_rate * loader_metadata["channels"]),
            "sample_rate": sample_rate,
            "bit_rate": loader_metadata["bit_rate"],
            "channels": loader_metadata["channels"],
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
