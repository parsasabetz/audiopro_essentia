"""Audio processing module."""

from .models import FrameFeatures
from .processors import process_frame
from .extractor import extract_features
from .audio_loader import load_and_preprocess_audio
from .validator import validate_audio_file, validate_audio_signal
from .frame_generator import create_frame_generator
from .spectrogram import compute_spectrogram, process_spectral_features


__all__ = [
    "FrameFeatures",
    "process_frame",
    "extract_features",
    "load_and_preprocess_audio",
    "validate_audio_file",
    "validate_audio_signal",
    "create_frame_generator",
    "compute_spectrogram",
    "process_spectral_features",
]
