"""
Audio Processing and Performance Monitoring Tool
"""

# Standard library imports
import json
import logging
import os
import threading
import time
from typing import List
import argparse

# Third-party imports
import librosa
import numpy as np
import psutil
import msgpack
import warnings  # Ensure warnings are imported

# Local imports
from .extractor import FRAME_LENGTH, HOP_LENGTH, extract_features
from .monitor import monitor_cpu_usage, print_performance_stats
from .metadata import get_file_metadata

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def analyze_audio(
    file_path: str, output_file: str, output_format: str = "json"
) -> None:
    """
    Main function to analyze audio and monitor performance.

    Args:
        file_path: Path to input audio file
        output_file: Path for output file (without extension)
        output_format: Format of the output file ('json' or 'msgpack')
    """
    # Single input validation block
    output_format = output_format.lower().strip()
    if output_format not in ["json", "msgpack"]:
        raise ValueError("output_format must be either 'json' or 'msgpack'")

    # Single extension handling
    final_output = f"{os.path.splitext(output_file)[0]}.{output_format}"
    logger.info(f"Output format: {output_format}")
    logger.info(f"Output file will be: {final_output}")

    start_time = time.time()
    cpu_usage_list: List[float] = []
    active_cores_list: List[int] = []
    stop_flag = threading.Event()

    # Start performance monitoring
    monitoring_thread = threading.Thread(
        target=monitor_cpu_usage,
        args=(
            psutil.Process(os.getpid()),
            cpu_usage_list,
            active_cores_list,
            stop_flag,
        ),
    )
    monitoring_thread.start()

    try:
        logger.info(f"Loading audio file: {file_path}")
        audio_data, sample_rate = librosa.load(file_path, sr=None)

        # Check if audio_data is not empty or silent
        if np.all(audio_data == 0) or len(audio_data) == 0:
            raise ValueError("Audio data is empty or silent.")

        # Calculate minimum required samples for pitch estimation
        min_samples = int(sample_rate * 0.1)  # At least 100ms of audio

        if len(audio_data) < min_samples:
            raise ValueError(
                f"Audio file too short. Minimum length required: {min_samples/sample_rate:.2f} seconds"
            )

        # Check signal energy
        signal_energy = np.sum(audio_data**2)
        if signal_energy < 1e-6:  # Arbitrary small threshold
            raise ValueError("Audio signal energy too low for analysis")

        # Calculate spectral flatness to assess frequency content
        spectral_flatness = librosa.feature.spectral_flatness(y=audio_data).mean()
        logger.info(f"Spectral Flatness: {spectral_flatness:.4f}")

        # Define a threshold for spectral flatness (e.g., 0.1)
        FLATNESS_THRESHOLD = 0.1
        if spectral_flatness > FLATNESS_THRESHOLD:
            logger.warning(
                "Audio has high spectral flatness. Pitch estimation may be unreliable."
            )

        # Check for sufficient frequency content
        spectral = np.abs(
            librosa.stft(audio_data, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
        )
        if np.all(spectral == 0):
            raise ValueError(
                "Audio data has insufficient frequency content for pitch estimation."
            )

        # Suppress specific librosa warnings temporarily
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            # Get file metadata from the new module
            metadata = get_file_metadata(file_path, audio_data, sample_rate)

            # Get tempo and beats with proper type conversion
            tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sample_rate)

        # Check if beats are detected to avoid empty frequency sets
        if len(beats) == 0:
            logger.warning("No beats detected in the audio. Skipping beat tracking.")
            tempo = 0.0
            beat_times = []
        else:
            tempo = float(tempo)  # Direct conversion without condition
            beat_times = librosa.frames_to_time(beats, sr=sample_rate).tolist()

        logger.info("Extracting features...")
        features = extract_features(audio_data, sample_rate)

        # Simplified analysis dictionary (removed redundant feature_map)
        analysis = {
            "metadata": metadata,
            "tempo": tempo,
            "beats": beat_times,
            "features": features,
        }

        # Save to file based on the specified format
        with open(final_output, "w" if output_format == "json" else "wb") as f:
            if output_format == "json":
                json.dump(analysis, f, indent=4)
            else:
                msgpack.pack(analysis, f)
        logger.info(f"Analysis saved to {final_output}")

        # Stop CPU monitoring
        stop_flag.set()
        monitoring_thread.join()

        end_time = time.time()
        print_performance_stats(start_time, end_time, cpu_usage_list, active_cores_list)

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        raise
    except Exception as e:
        stop_flag.set()
        monitoring_thread.join()
        logger.error(f"Error analyzing audio: {str(e)}")
        raise

    finally:
        if not stop_flag.is_set():
            stop_flag.set()
            monitoring_thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze audio files.")
    parser.add_argument("input_file", type=str, help="Path to input audio file")
    parser.add_argument(
        "output_file", type=str, help="Name for output file without extension"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "msgpack"],
        default="json",
        help="Output format: 'json' (default) or 'msgpack'",
    )
    args = parser.parse_args()

    # Simplified extension handling
    output_file = f"{args.output_file}.{args.format}"
    analyze_audio(args.input_file, output_file, output_format=args.format)
