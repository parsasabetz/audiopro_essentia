# typing imports
from typing import Optional, Tuple

# Standard library imports
from pathlib import Path
import mimetypes
import hashlib
import numpy as np

# Third-party scientific/audio processing libraries
import torch
import torchaudio
import ffmpeg

# Local application imports
from audiopro.utils.logger import get_logger
from audiopro.output.types import LoaderMetadata, TimeRange
from audiopro.errors.exceptions import AudioIOError, AudioValidationError
from .validator import validate_audio_file, validate_audio_signal

# Configure logging for this module
logger = get_logger(__name__)


def load_and_preprocess_audio(
    file_path: str, time_range: Optional[TimeRange] = None
) -> Tuple[np.ndarray, int, LoaderMetadata, float]:
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
            - loader_metadata (LoaderMetadata): Metadata about the loaded audio file
            - duration (float): Duration of the audio in seconds

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
        validate_audio_file(file_path)

        logger.info("Loading audio file: %s", file_path)
        try:
            # Force channels_first=False to get waveform shape (time, channels)
            waveform, sample_rate = torchaudio.load(file_path, channels_first=False)

            # Get audio info using ffmpeg
            probe = ffmpeg.probe(file_path)
            format_info = probe["format"]
            stream_info = probe["streams"][
                0
            ]  # Assuming the first stream is the audio stream
            bit_rate = int(format_info.get("bit_rate", 0))
            codec = stream_info.get("codec_name", "UNKNOWN")
            original_channels = int(stream_info.get("channels", 1))

            # Log extracted metadata for debugging
            logger.info(
                f"Extracted metadata: bit_rate={bit_rate}, codec={codec}, channels={original_channels}"
            )

            # Calculate MD5 hash of the file
            with open(file_path, "rb") as f:
                md5 = hashlib.md5(f.read()).hexdigest()

            # Convert to mono by averaging channels if necessary
            if waveform.ndim > 1 and waveform.shape[1] > 1:
                waveform = torch.mean(waveform, dim=1)

            # Convert to numpy array
            audio_data = waveform.numpy()

        except Exception as e:
            raise AudioIOError(
                message="File appears corrupted or invalid audio format",
                filepath=file_path,
                operation="read",
                error=str(e),
            ) from e

        if time_range:
            start_sample = int(time_range.get("start", 0) * sample_rate)
            end_sample = int(
                time_range.get("end", len(audio_data) / sample_rate) * sample_rate
            )
            end_sample = min(end_sample, len(audio_data))
            audio_data = audio_data[start_sample:end_sample]
            logger.info(
                f"Sliced audio from {start_sample / sample_rate:.2f}s to {end_sample / sample_rate:.2f}s"
            )

        # Drop last sample once if odd-length
        if len(audio_data) % 2 != 0:
            audio_data = audio_data[:-1]
            logger.info("Dropped last sample to enforce even length for FFT.")

        validate_audio_signal(audio_data, sample_rate)

        # Precompute file metadata once
        path_obj = Path(file_path)
        file_stats = path_obj.stat()
        loader_metadata: LoaderMetadata = {
            "filename": path_obj.name,
            "format": path_obj.suffix[1:],
            "size_mb": file_stats.st_size / (1024**2),
            "created_date": file_stats.st_ctime,
            "mime_type": mimetypes.guess_type(file_path)[0] or "unknown",
            "md5_hash": md5,
            "bit_rate": bit_rate,
            "codec": codec,
            "channels": original_channels,  # use original channel count in metadata
            "sample_rate": sample_rate,
        }

        signal_energy = np.sum(audio_data**2)
        if signal_energy < 1e-6:
            raise AudioValidationError(
                message="Invalid audio content",
                parameter="signal_energy",
                expected=">1e-6",
                actual=signal_energy,
            )

        min_samples = int(sample_rate * 0.1)
        if len(audio_data) < min_samples:
            raise ValueError(
                f"Audio file too short. Minimum length required: {min_samples / sample_rate:.2f} seconds"
            )

        duration = len(audio_data) / sample_rate
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
            suggestion="Try checking file permissions and disk space",
        ) from e
