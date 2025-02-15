# typing imports
from functools import lru_cache

# Third-party imports
import torchaudio.transforms as T
import torch

# Local application imports
from audiopro.utils.constants import (  # pylint: disable=no-name-in-module
    FRAME_LENGTH,
    HOP_LENGTH,
)

# Pre-compute constants for efficiency
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=8)
def get_transforms(sample_rate: int):
    """
    Generates and returns a Spectrogram and MFCC transformation.

    Args:
        sample_rate (int): The sample rate of the audio.

    Returns:
        tuple[torch.nn.Module, torch.nn.Module]: A tuple containing the Spectrogram and MFCC transformations.
    """

    spectrum_transform = T.Spectrogram(
        n_fft=FRAME_LENGTH,
        win_length=FRAME_LENGTH,  # Match window length with FFT size
        hop_length=HOP_LENGTH,
        pad=0,
        window_fn=torch.hann_window,  # Use default hann window
        power=2.0,  # Use power spectrum
        normalized=False,
        wkwargs=None,
        center=True,
        pad_mode="reflect",
        onesided=True,
    ).to(DEVICE)

    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,
        melkwargs={
            "n_fft": FRAME_LENGTH,
            "win_length": FRAME_LENGTH,  # Match window length with FFT size
            "hop_length": HOP_LENGTH,
            "n_mels": 40,
            "mel_scale": "htk",
            "normalized": True,
            "center": True,
            "pad_mode": "reflect",
            "power": 2.0,
            "window_fn": torch.hann_window,  # Use default hann window
        },
    ).to(DEVICE)

    return spectrum_transform, mfcc_transform
