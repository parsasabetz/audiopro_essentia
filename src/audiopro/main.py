"""
Audio Processing and Performance Monitoring Tool
"""

# Standard library imports
import logging
import os
import threading
import time
from typing import List
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Local imports
# CLI parsing
from .arg_parser import parse_arguments

# Audio processing
from .audio.audio_loader import load_and_preprocess_audio
from .audio.extractor import extract_features
from .audio.metadata import get_file_metadata

# Analysis utilities
from .utils import optimized_convert_to_native_types, extract_rhythm

# Performance monitoring
from .monitor.monitor import monitor_cpu_usage, print_performance_stats

# Output handling
from .output.output_handler import write_output
from .output.types import AudioAnalysis

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


async def analyze_audio(
    file_path: str,
    output_file: str,
    output_format: str = "msgpack",
    skip_monitoring: bool = False,
) -> None:
    """
    Main function to analyze audio and monitor performance.

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
        logger.info("Starting audio analysis pipeline...")
        logger.info(f"Loading audio file: {file_path}")
        audio_data, sample_rate = load_and_preprocess_audio(file_path)
        logger.info(f"Audio loaded successfully. Sample rate: {sample_rate}Hz")

        with ThreadPoolExecutor() as executor:
            logger.info("Submitting parallel processing tasks...")

            # Submit tasks with logging
            logger.info("Extracting metadata...")
            metadata_future = executor.submit(
                get_file_metadata, file_path, audio_data, sample_rate
            )

            logger.info("Extracting audio features...")
            features_future = executor.submit(extract_features, audio_data, sample_rate)

            logger.info("Analyzing rhythm patterns...")
            rhythm_future = executor.submit(extract_rhythm, audio_data)

            # Get results with logging
            logger.info("Waiting for task completion...")
            metadata = metadata_future.result()
            logger.info("Metadata extraction completed")

            features = features_future.result()
            logger.info("Feature extraction completed")

            tempo, beat_positions = rhythm_future.result()
            logger.info("Rhythm analysis completed")

            beat_times = beat_positions.tolist()
            logger.info(f"Found {len(beat_times)} beats, tempo: {tempo:.2f} BPM")

        # Check if beats are detected to avoid empty frequency sets
        if not beat_times:
            logger.warning("No beats detected in the audio. Skipping beat tracking.")
            tempo = 0.0

        logger.info("Compiling analysis results...")
        analysis: AudioAnalysis = {
            "metadata": metadata,
            "tempo": tempo,
            "beats": beat_times,
            "features": features,
        }

        # Convert all numpy types to native Python types before serialization
        logger.info("Converting data types for output...")
        analysis = optimized_convert_to_native_types(analysis)

        logger.info(f"Writing output to {final_output}...")
        await write_output(analysis, final_output, output_format)
        logger.info("Output written successfully")

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
        logger.error(f"Analysis failed: {str(e)}")
        if stop_flag and monitoring_thread:
            stop_flag.set()
            monitoring_thread.join()
        raise

    finally:
        logger.info("Cleaning up resources...")
        if stop_flag and not stop_flag.is_set():
            stop_flag.set()
            if monitoring_thread:
                monitoring_thread.join()


if __name__ == "__main__":
    args = parse_arguments()
    output_file = f"{args.output_file}.{args.format}"
    asyncio.run(
        analyze_audio(
            args.input_file,
            output_file,
            output_format=args.format,
            skip_monitoring=args.skip_monitoring,
        )
    )
