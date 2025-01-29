"""
Audio Processing and Performance Monitoring Tool
"""

# Standard library imports
import logging
import os
import threading
import time
from typing import List
import asyncio

# Third-party imports
import warnings

# Local imports
from .arg_parser import parse_arguments
from .audio.audio_loader import load_and_preprocess_audio
from .audio.extractor import extract_features
from .utils import optimized_convert_to_native_types, extract_rhythm
from .audio.metadata import get_file_metadata
from .monitor.monitor import monitor_cpu_usage, print_performance_stats
from .output.output_handler import write_output

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
        audio_data, sample_rate = load_and_preprocess_audio(file_path)

        # Suppress specific warnings temporarily
        with warnings.catch_warnings():
            # Get file metadata from the new module
            metadata = await get_file_metadata(file_path, audio_data, sample_rate)

            # Extract rhythm features
            tempo, beat_positions = extract_rhythm(audio_data)
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

        await write_output(analysis, final_output, output_format)

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
    args = parse_arguments()

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
