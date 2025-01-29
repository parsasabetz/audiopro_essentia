"""
Audio Processing and Performance Monitoring Tool
"""

# Standard library imports
import logging
import os
import threading
import time
from typing import List
import argparse
import asyncio

# Third-party imports
import essentia.standard as es
import numpy as np
import msgpack
import warnings
import aiofiles
import orjson  # Use orjson for faster JSON serialization

# Local imports
from .extractor import extract_features, optimized_convert_to_native_types
from .monitor import monitor_cpu_usage, print_performance_stats
from .metadata import get_file_metadata

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


async def analyze_audio(
    file_path: str,
    output_file: str,
    output_format: str = "msgpack",  # Changed default from "json" to "msgpack"
    skip_monitoring: bool = False,
) -> None:
    """
    Main function to analyze audio and monitor performance asynchronously.

    Args:
        file_path: Path to input audio file
        output_file: Path for output file (without extension)
        output_format: Format of the output file ('msgpack' by default or 'json' if specified)
        skip_monitoring: Flag to skip performance monitoring
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

    if not skip_monitoring:
        stop_flag = threading.Event()
        monitoring_thread = threading.Thread(
            target=monitor_cpu_usage,
            args=(
                cpu_usage_list,
                active_cores_list,
                stop_flag,
            ),
        )
        monitoring_thread.start()
    else:
        stop_flag = None
        monitoring_thread = None

    try:
        logger.info(f"Loading audio file: {file_path}")
        loader = es.MonoLoader(filename=file_path)

        # If streaming isn't possible, keep as is:
        audio_data = loader()
        sample_rate = int(loader.paramValue("sampleRate"))

        # Ensure audio_data has an even length
        if len(audio_data) % 2 != 0:
            audio_data = np.append(audio_data, 0.0).astype(np.float32)
            logger.info("Appended zero to make audio_data length even for FFT.")

        # Check if audio_data is not empty or silent
        if not np.any(audio_data):
            raise ValueError("Audio data is empty or silent.")

        # Calculate minimum required samples for pitch estimation
        min_samples = int(sample_rate * 0.1)  # At least 100ms of audio

        if len(audio_data) < min_samples:
            raise ValueError(
                f"Audio file too short. Minimum length required: {min_samples/sample_rate:.2f} seconds"
            )

        # Check signal energy
        signal_energy = np.sum(audio_data**2)
        if signal_energy < 1e-6:
            raise ValueError("Audio signal energy too low for analysis")

        # Suppress specific warnings temporarily
        with warnings.catch_warnings():
            # Get file metadata from the new module
            metadata = await get_file_metadata(file_path, audio_data, sample_rate)

            # Replace tempo and beat tracking with RhythmExtractor2013
            rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
            tempo, beat_positions, _, _, _ = rhythm_extractor(audio_data)
            beat_times = beat_positions.tolist()

        # Check if beats are detected to avoid empty frequency sets
        if not beat_times:
            logger.warning("No beats detected in the audio. Skipping beat tracking.")
            tempo = 0.0

        logger.info("Extracting features...")
        features = extract_features(audio_data, sample_rate)

        # Analysis dictionary
        analysis = {
            "metadata": metadata,
            "tempo": tempo,
            "beats": beat_times,
            "features": features,
        }

        # Convert all numpy types to native Python types before serialization
        analysis = optimized_convert_to_native_types(analysis)

        # Use asynchronous file writing
        if output_format == "json":
            async with aiofiles.open(final_output, "w") as f:
                await f.write(orjson.dumps(analysis).decode())
        else:
            # Serialize the analysis dictionary to MessagePack bytes
            packed_data = msgpack.packb(analysis)
            async with aiofiles.open(final_output, "wb") as f:
                await f.write(packed_data)

        logger.info(f"Analysis saved to {final_output}")

        # Stop CPU monitoring
        if stop_flag and monitoring_thread:
            stop_flag.set()
            monitoring_thread.join()

        end_time = time.time()
        print_performance_stats(start_time, end_time, cpu_usage_list, active_cores_list)

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        raise
    except Exception as e:
        if stop_flag and monitoring_thread:
            stop_flag.set()
            monitoring_thread.join()
        logger.error(f"Error analyzing audio: {str(e)}")
        raise

    finally:
        if stop_flag and not stop_flag.is_set():
            stop_flag.set()
            if monitoring_thread:
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
        default="msgpack",  # Changed default from "json" to "msgpack"
        help="Output format: 'msgpack' (default) or 'json'",
    )
    parser.add_argument(
        "--skip-monitoring",
        action="store_true",
        help="Skip performance monitoring to reduce overhead",
    )
    args = parser.parse_args()

    # Simplified extension handling
    output_file = f"{args.output_file}.{args.format}"
    asyncio.run(
        analyze_audio(
            args.input_file,
            output_file,
            output_format=args.format,
            skip_monitoring=args.skip_monitoring,
        )
    )
