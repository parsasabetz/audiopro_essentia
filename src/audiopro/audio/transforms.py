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
def get_transforms(sample_rate: int, compute_spectrum: bool, compute_mfcc: bool):
    """
    Generates and returns the required transformations based on the feature configuration.

    Args:
        sample_rate (int): The sample rate of the audio.
        compute_spectrum (bool): Whether to compute the spectrogram transformation.
        compute_mfcc (bool): Whether to compute the MFCC transformation.

    Returns:
        tuple: A tuple containing the requested transformations. If a transformation is not requested, None is returned in its place.
    """

    spectrum_transform = None
    mfcc_transform = None

    if compute_spectrum:
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

    if compute_mfcc:
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
