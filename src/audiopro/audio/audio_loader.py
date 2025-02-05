# Standard library imports
import os
import mimetypes
from pathlib import Path
from typing import Optional

# Third-party scientific/audio processing libraries
import numpy as np
import essentia.standard as es

# Local application imports
from audiopro.utils.logger import get_logger
from audiopro.output.types import LoaderMetadata, TimeRange
from audiopro.errors.exceptions import AudioIOError, AudioValidationError

# Configure logging for this module
logger = get_logger(__name__)


def load_and_preprocess_audio(
    file_path: str, time_range: Optional[TimeRange] = None
) -> tuple:
    """Loads and preprocesses an audio file for analysis.

    This function loads an audio file, converts it to mono, and performs several
    preprocessing checks to ensure the audio data is suitable for analysis.

    Args:
        file_path (str): Path to the audio file to be loaded.
        time_range (Optional[TimeRange]): Time range as a dictionary with 'start' and 'end' keys.
            If 'end' is None, processes until the end of file.
            If 'start' is None, processes from the beginning.

    Returns:
        tuple: A tuple containing:
            - audio_data (numpy.ndarray): Preprocessed audio samples as a numpy array
            - sample_rate (int): Sample rate of the audio in Hz
            - duration (float): Duration of the trimmed audio in seconds

    Raises:
        AudioIOError: If there is an error loading or processing the audio file.
        ValueError: If the audio is:
        AudioValidationError: If the audio has insufficient signal energy (< 1e-6)
            - Empty or silent
            - Too short (less than 100ms)
            - Has insufficient signal energy (< 1e-6)

    Notes:
        - If the audio data length is odd, a zero is appended to make it even for FFT
        - The audio is converted to mono during loading
        - Minimum audio length required is 100ms
    """
    try:
        logger.info("Loading audio file: %s", file_path)
        try:
            audio_data, sample_rate, channels, md5, bit_rate, codec = es.AudioLoader(
                filename=file_path, computeMD5=True
            )()
        except Exception as e:
            raise AudioIOError(
                message="Failed to load audio file",
                filepath=file_path,
                operation="read",
                error=str(e),
            ) from e

        if time_range:
            start_sample = int(time_range.get("start", 0) * sample_rate)
            end_sample = (
                int(time_range["end"] * sample_rate)
                if "end" in time_range
                else len(audio_data)
            )
            end_sample = min(end_sample, len(audio_data))

            audio_data = audio_data[start_sample:end_sample]
            logger.info(
                f"Sliced audio from {start_sample/sample_rate:.2f}s to {end_sample/sample_rate:.2f}s"
            )

        duration = len(audio_data) / sample_rate

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

        # In-place even-length adjustment: instead of padding, drop the last sample if odd.
        if len(audio_data) % 2 != 0:
            audio_data = audio_data[:-1]  # Drop one sample to avoid extra allocation.
            logger.info("Dropped last sample to enforce even length for FFT.")

        # Combine empty/silent check with signal energy check to reduce redundancy
        signal_energy = np.sum(audio_data**2)
        if not np.any(audio_data) or signal_energy < 1e-6:
            raise AudioValidationError(
                message="Invalid audio content",
                parameter="signal_energy",
                expected=">1e-6",
                actual=signal_energy,
            )

        # Calculate minimum required samples for pitch estimation
        min_samples = int(sample_rate * 0.1)  # At least 100ms of audio

        if len(audio_data) < min_samples:
            raise ValueError(
                f"Audio file too short. Minimum length required: {min_samples/sample_rate:.2f} seconds"
            )

        logger.info("Audio loaded successfully. Sample rate: %dHz", sample_rate)
        return audio_data, sample_rate, loader_metadata, duration

    except (AudioIOError, AudioValidationError):
        raise
    except Exception as e:
        raise AudioIOError(
            message="Unexpected error during audio loading",
            filepath=file_path,
            operation="process",
            error=str(e),
        ) from e
