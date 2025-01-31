# Third-party imports
import essentia.standard as es
import numpy as np

# Local imports
from audiopro.utils.logger import get_logger

# Configure logging
logger = get_logger(__name__)


def load_and_preprocess_audio(file_path: str):
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
    loader = es.MonoLoader(filename=file_path)

    audio_data = loader()
    sample_rate = int(loader.paramValue("sampleRate"))

    # Ensure audio_data has an even length
    if len(audio_data) % 2 != 0:
        audio_data = np.append(audio_data, 0.0).astype(np.float32)
        logger.info("Appended zero to make audio_data length even for FFT.")

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

    return audio_data, sample_rate
