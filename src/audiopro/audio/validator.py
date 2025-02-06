"""Audio file validation utilities."""

# typing imports
from typing import Set

# Standard library imports
from pathlib import Path
import mimetypes

# Third-party imports
import numpy as np

# Local application imports
from audiopro.errors.exceptions import AudioIOError, AudioValidationError

# Audio format constraints
SUPPORTED_FORMATS: Set[str] = frozenset(
    {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aiff"}
)
MAX_FILE_SIZE_MB: int = 50
MIN_SIGNAL_ENERGY: float = 1e-6


def validate_audio_file(file_path: str) -> None:
    """
    Validates an audio file based on its path.

    This function performs several checks to ensure the audio file is valid:
    1. Checks if the file exists.
    2. Checks if the path is a file.
    3. Validates the file format against supported formats.
    4. Checks if the file size is within the allowed limit.
    5. Validates the MIME type to ensure it is an audio file.

    Args:
        file_path (str): The path to the audio file to be validated.

    Raises:
        AudioIOError: If any of the validation checks fail, an AudioIOError is raised with a relevant message.
    """

    path = Path(file_path)

    if not path.exists() or not path.is_file():
        raise AudioIOError(
            message="File does not exist or path is not a file",
            filepath=file_path,
            operation="validate",
        )

    ext = path.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise AudioIOError(
            message=f"Unsupported audio format. Supported: {', '.join(SUPPORTED_FORMATS)}",
            filepath=file_path,
            operation="validate",
            format=ext,
        )

    stat_result = path.stat()
    file_size_mb = stat_result.st_size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise AudioIOError(
            message=f"File size exceeds {MAX_FILE_SIZE_MB}MB",
            filepath=file_path,
            operation="validate",
            size_mb=file_size_mb,
        )

    mime_type = mimetypes.guess_type(file_path)[0]
    if mime_type and not mime_type.startswith("audio/"):
        raise AudioIOError(
            message="File MIME type is not audio",
            filepath=file_path,
            operation="validate",
            mime_type=mime_type,
        )


def validate_audio_signal(audio_data: np.ndarray, sample_rate: int) -> None:
    """
    Validates the given audio signal based on several criteria.

    Parameters:
        audio_data (np.ndarray): The audio signal data as a NumPy array.
        sample_rate (int): The sample rate of the audio signal in Hz.

    Raises:
        AudioValidationError: If the audio signal contains invalid values (NaN or Inf).
        AudioValidationError: If the audio signal energy is below the minimum threshold.
        AudioValidationError: If the audio file is too short (less than 100ms).

    Notes:
        - The function checks if the audio data contains any NaN or Inf values.
        - It calculates the signal energy and compares it against a predefined minimum threshold.
        - It ensures the audio duration is at least 100 milliseconds.
    """

    if not np.all(np.isfinite(audio_data)):
        raise AudioValidationError(
            message="Audio contains invalid values (NaN or Inf)",
            parameter="signal_validity",
            actual="Contains NaN/Inf values",
        )

    signal_energy = np.sum(audio_data**2)
    if signal_energy < MIN_SIGNAL_ENERGY:
        raise AudioValidationError(
            message="Audio signal too weak",
            parameter="signal_energy",
            expected=f">{MIN_SIGNAL_ENERGY}",
            actual=signal_energy,
            suggestion="Check if file is corrupted or contains only silence",
        )

    min_samples = int(sample_rate * 0.1)  # 100ms minimum
    if len(audio_data) < min_samples:
        raise AudioValidationError(
            message="Audio file too short",
            parameter="duration",
            expected=f">={min_samples/sample_rate:.2f}s",
            actual=f"{len(audio_data)/sample_rate:.2f}s",
        )
