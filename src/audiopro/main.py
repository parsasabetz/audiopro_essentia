"""
Audio Processing and Performance Monitoring Tool
"""

# Standard library imports
import os
import threading
import time
from typing import List
from concurrent.futures import ThreadPoolExecutor
import asyncio
import signal
from contextlib import contextmanager

# Local imports
# CLI parsing
from .arg_parser import parse_arguments

# Audio processing
from .audio.audio_loader import load_and_preprocess_audio
from .audio.extractor import extract_features
from .audio.metadata import get_file_metadata

# Analysis utilities
from .utils import optimized_convert_to_native_types, extract_rhythm
from .utils.logger import get_logger

# Performance monitoring
from .monitor.monitor import monitor_cpu_usage, print_performance_stats

# Output handling
from .output.output_handler import write_output
from .output.types import AudioAnalysis

logger = get_logger(__name__)


@contextmanager
def graceful_shutdown():
    """Context manager for graceful shutdown."""
    original_handlers = {}
    stop_flag = threading.Event()

    def handler(_signum, _frame):
        logger.info("Received shutdown signal, cleaning up...")
        stop_flag.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        original_handlers[sig] = signal.signal(sig, handler)

    try:
        yield stop_flag
    finally:
        for sig, original_handler in original_handlers.items():
            signal.signal(sig, original_handler)


async def analyze_audio(
    file_path: str,
    output_path: str,
    output_format: str = "msgpack",
    skip_monitoring: bool = False,
) -> None:
    """
    Main function to analyze audio and monitor performance.

    Args:
        file_path: Path to input audio file
        output_path: Path for output file (without extension)
        output_format: Format of the output file ('msgpack' by default or 'json' if specified)
        skip_monitoring: Flag to skip performance monitoring
    """
    # Single input validation block
    output_format = output_format.lower().strip()
    if output_format not in ["json", "msgpack"]:
        raise ValueError("output_format must be either 'json' or 'msgpack'")

    # Single extension handling (without logging)
    final_output = f"{os.path.splitext(output_path)[0]}.{output_format}"

    start_time = time.time()
    cpu_usage_list: List[float] = []
    active_cores_list: List[int] = []

    with graceful_shutdown() as stop_flag:
        monitoring_thread = None
        try:
            if not skip_monitoring:
                monitoring_thread = threading.Thread(
                    target=monitor_cpu_usage,
                    args=(cpu_usage_list, active_cores_list, stop_flag),
                    daemon=True,  # Make thread daemon so it exits with main thread
                )
                monitoring_thread.start()

            logger.info("Starting audio analysis pipeline...")
            audio_data, sample_rate = load_and_preprocess_audio(file_path)

            with ThreadPoolExecutor() as executor:
                logger.info("Submitting parallel processing tasks...")

                # Submit tasks with logging
                logger.info("Extracting metadata...")
                metadata_future = executor.submit(
                    get_file_metadata, file_path, audio_data, sample_rate
                )

                logger.info("Extracting audio features...")
                features_future = executor.submit(
                    extract_features, audio_data, sample_rate
                )

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
                logger.info("Found %d beats, tempo: %.2f BPM", len(beat_times), tempo)

            # Check if beats are detected to avoid empty frequency sets
            if not beat_times:
                logger.warning(
                    "No beats detected in the audio. Skipping beat tracking."
                )
                tempo = 0.0

            # Compile analysis results
            logger.info("Compiling analysis results...")
            analysis: AudioAnalysis = optimized_convert_to_native_types(
                {
                    "metadata": metadata,
                    "tempo": tempo,
                    "beats": beat_times,
                    "features": features,
                }
            )

            # Stop monitoring before writing output
            if monitoring_thread:
                stop_flag.set()
                monitoring_thread.join(timeout=2)  # Wait max 2 seconds

            await write_output(analysis, final_output, output_format)

            end_time = time.time()
            print_performance_stats(
                start_time, end_time, cpu_usage_list, active_cores_list
            )

        except Exception as e:
            logger.error("Analysis failed: %s", str(e))
            raise
        finally:
            # Ensure monitoring thread is stopped
            if stop_flag and not stop_flag.is_set():
                stop_flag.set()
            if monitoring_thread and monitoring_thread.is_alive():
                monitoring_thread.join(timeout=1)


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
