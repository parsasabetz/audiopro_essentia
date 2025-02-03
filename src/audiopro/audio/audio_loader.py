# Standard library imports
import os
import mimetypes
from pathlib import Path

# Third-party scientific/audio processing libraries
import numpy as np
import essentia.standard as es

# Local application imports
from audiopro.utils.logger import get_logger
from audiopro.output.types import LoaderMetadata

# Configure logging for this module
logger = get_logger(__name__)


def load_and_preprocess_audio(file_path: str) -> tuple:
    """Loads and preprocesses an audio file for analysis.

    This function loads an audio file, converts it to mono, and performs several
    preprocessing checks to ensure the audio data is suitable for analysis.

    Args:
        file_path (str): Path to the audio file to be loaded.

    Returns:
        tuple: A tuple containing:
            - audio_data (numpy.ndarray): Preprocessed audio samples as a numpy array
            - sample_rate (int): Sample rate of the audio in Hz

    Raises:
        ValueError: If the audio is:
            - Empty or silent
            - Too short (less than 100ms)
            - Has insufficient signal energy (< 1e-6)

    Notes:
        - If the audio data length is odd, a zero is appended to make it even for FFT
        - The audio is converted to mono during loading
        - Minimum audio length required is 100ms
    """
    logger.info("Loading audio file: %s", file_path)
    audio_data, sample_rate, channels, md5, bit_rate, codec = es.AudioLoader(
        filename=file_path
    )()

    # Compute file stats once and pack extra metadata
    file_stats = os.stat(file_path)
    loader_metadata: LoaderMetadata = {  # annotated with LoaderMetadata type
        "filename": Path(file_path).name,
        "format": Path(file_path).suffix[1:],
        "size_mb": file_stats.st_size / (1024**2),
        "created_date": file_stats.st_ctime,
        "mime_type": mimetypes.guess_type(file_path)[0] or "unknown",
        "md5_hash": md5,
        "bit_rate": bit_rate,
        "codec": codec,
        "channels": channels,
        "sample_rate": sample_rate,
    }

    # In-place even-length adjustment to minimize extra copy creation
    if len(audio_data) % 2 != 0:
        audio_data = np.pad(audio_data, (0, 1), mode='constant').astype(np.float32)
        logger.info("Padded audio_data to even length for FFT.")

    # Combine empty/silent check with signal energy check to reduce redundancy
    signal_energy = np.sum(audio_data**2)
    if not np.any(audio_data) or signal_energy < 1e-6:
        logger.error("Audio data is empty, silent, or has insufficient energy.")
        raise ValueError("Audio data is empty, silent, or has insufficient energy.")

    # Calculate minimum required samples for pitch estimation
    min_samples = int(sample_rate * 0.1)  # At least 100ms of audio

    if len(audio_data) < min_samples:
        raise ValueError(
            f"Audio file too short. Minimum length required: {min_samples/sample_rate:.2f} seconds"
        )

    logger.info("Audio loaded successfully. Sample rate: %dHz", sample_rate)
    return audio_data, sample_rate, loader_metadata
