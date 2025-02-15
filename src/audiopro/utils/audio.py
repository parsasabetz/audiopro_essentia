# typing imports
from typing import Tuple
from functools import lru_cache

# third-party imports
import torch
import numpy as np
import torchaudio.transforms as T

# local imports
from audiopro.utils.logger import get_logger
from audiopro.utils.constants import HOP_LENGTH

# setup logger
logger = get_logger(__name__)


class BeatTrackerSingleton:
    """
    BeatTrackerSingleton is a thread-safe singleton class for beat tracking.

    This class ensures that only one instance of the beat tracker is created and used throughout the application.
    It uses TorchAudio's beat detection functionality.

    Attributes:
        _instance (BeatTrackerSingleton): The singleton instance of the class.
        device (torch.device): The device to run computations on (CPU/CUDA).
        sample_rate (int): The sample rate used for beat tracking.

    Methods:
        __new__(cls): Creates and returns the singleton instance of the class.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            cls._instance.sample_rate = 44100  # Standard sample rate for beat tracking
        return cls._instance


def extract_rhythm(audio: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Extracts the rhythm from an audio signal using onset strength signal and dynamic programming.

    Parameters:
    audio (np.ndarray): A numpy array containing the audio signal.

    Returns:
    Tuple[float, np.ndarray]: A tuple containing the tempo (in beats per minute)
                              and an array of beat positions.

    Raises:
    TypeError: If the input audio is not a numpy array.
    ValueError: If the input audio data is empty.
    RuntimeError: If rhythm extraction fails for any reason.
    """

    # Use numpy's built-in type checking (faster than isinstance)
    if not hasattr(audio, "dtype") or not hasattr(audio, "size"):
        raise TypeError("Audio must be a numpy array")
    if audio.size == 0:
        raise ValueError("Audio data is empty")

    try:
        # Convert to mono if needed and ensure float32 type
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1, dtype=np.float32)
        else:
            audio = audio.astype(np.float32, copy=False)

        # Get singleton instance for device management
        tracker = BeatTrackerSingleton()

        # Convert numpy array to torch tensor
        audio_tensor = torch.from_numpy(audio).to(tracker.device)

        # Ensure audio is the right shape (1, n_samples)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Create mel spectrogram transform
        mel_transform = T.MelSpectrogram(
            sample_rate=tracker.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=20,
            f_max=tracker.sample_rate // 2,
            power=2.0,  # Use power spectrogram for better onset detection
        ).to(tracker.device)

        # Compute mel spectrogram
        mel_spec = mel_transform(audio_tensor)

        # Convert to dB scale and normalize
        mel_spec = torch.log1p(mel_spec)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

        # Compute onset strength signal
        onset_env = torch.diff(mel_spec, dim=2)
        onset_env = torch.clamp(onset_env, min=0)
        onset_env = torch.sum(onset_env, dim=1).squeeze()

        # Convert to numpy for peak picking
        onset_env = onset_env.cpu().numpy()

        # Normalize onset envelope
        onset_env = (onset_env - onset_env.mean()) / (onset_env.std() + 1e-8)

        # Parameters for beat tracking
        frame_length = int(
            0.1 * tracker.sample_rate / HOP_LENGTH
        )  # 100ms window using our hop length
        hop_length = HOP_LENGTH  # Use HOP_LENGTH constant instead of hardcoded value

        # Compute autocorrelation of onset envelope
        def autocorr(x):
            result = np.correlate(x, x, mode="full")
            return result[result.size // 2 :]

        # Get autocorrelation of onset envelope
        ac = autocorr(onset_env)

        # Find peaks in autocorrelation
        # Look for peaks in reasonable tempo range (60-180 BPM)
        min_lag = int(60 * tracker.sample_rate / (hop_length * 180))  # 180 BPM
        max_lag = int(60 * tracker.sample_rate / (hop_length * 60))  # 60 BPM

        ac_peaks = []
        for i in range(min_lag, min(len(ac), max_lag)):
            if ac[i] > ac[i - 1] and ac[i] > ac[i + 1]:
                ac_peaks.append((ac[i], i))

        if not ac_peaks:
            return 120.0, np.array([])

        # Sort peaks by correlation value
        ac_peaks.sort(reverse=True)

        # Get tempo from highest peak
        tempo_lag = ac_peaks[0][1]
        tempo = 60 * tracker.sample_rate / (hop_length * tempo_lag)

        # Find peaks in onset envelope for beat positions
        peaks = []
        threshold = np.percentile(onset_env, 80)  # Adaptive threshold

        for i in range(frame_length, len(onset_env) - frame_length):
            if onset_env[i] > threshold:
                if onset_env[i] == max(onset_env[i - frame_length : i + frame_length]):
                    peaks.append(i)

        if not peaks:
            return tempo, np.array([])

        # Convert peak positions to time
        beat_positions = np.array(peaks) * hop_length / tracker.sample_rate

        # Regularize beat positions using tempo
        if len(beat_positions) > 1:
            # Expected beat interval
            beat_period = 60.0 / tempo

            # Adjust beat positions to be more regular
            regular_beats = []
            current_beat = beat_positions[0]
            regular_beats.append(current_beat)

            for beat in beat_positions[1:]:
                expected_next_beat = current_beat + beat_period
                # If this beat is close to where we expect it, use it
                if abs(beat - expected_next_beat) < 0.1:  # 100ms tolerance
                    current_beat = beat
                    regular_beats.append(current_beat)
                # If we've missed a beat, interpolate
                elif beat - current_beat > 1.5 * beat_period:
                    while current_beat + beat_period < beat:
                        current_beat += beat_period
                        regular_beats.append(current_beat)
                    current_beat = beat
                    regular_beats.append(current_beat)

            beat_positions = np.array(regular_beats)

        return tempo, beat_positions

    except Exception as e:
        logger.error(f"Rhythm extraction failed: {str(e)}")
        raise RuntimeError(f"Failed to extract rhythm: {str(e)}") from e


@lru_cache(maxsize=32)
def compute_spectral_bandwidth(
    spectrum: np.ndarray, freqs: np.ndarray, centroid: float
) -> float:
    """
    Compute the spectral bandwidth of a given spectrum using TorchAudio.

    Spectral bandwidth is a measure of the width of the spectrum around its centroid.

    Parameters:
        spectrum (np.ndarray): The amplitude spectrum of the signal.
        freqs (np.ndarray): The corresponding frequencies of the spectrum.
        centroid (float): The spectral centroid of the spectrum.

    Returns:
        float: The spectral bandwidth of the spectrum.

    Raises:
        TypeError: If spectrum or freqs are not numpy arrays.
        ValueError: If spectrum and freqs do not have the same size.
    """

    if not (isinstance(spectrum, np.ndarray) and isinstance(freqs, np.ndarray)):
        raise TypeError("Spectrum and frequencies must be numpy arrays")

    if spectrum.size != freqs.size:
        raise ValueError("Spectrum and frequencies must have the same size")

    # Pre-calculate condition to avoid unnecessary computation
    if np.sum(spectrum) <= 1e-10:
        return 0.0

    # Convert to torch tensors for computation
    spectrum_tensor = torch.from_numpy(spectrum)
    freqs_tensor = torch.from_numpy(freqs)

    # Calculate bandwidth using broadcasting
    freq_diff = freqs_tensor - centroid
    variance = torch.sum(freq_diff.pow(2) * spectrum_tensor) / (
        torch.sum(spectrum_tensor) + 1e-10
    )

    return float(torch.sqrt(torch.clamp(variance, min=0.0)).item())
