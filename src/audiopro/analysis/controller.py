"""
Audio analysis controller module.
"""

# typing imports
from typing import Optional

# Standard library imports
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# Local application imports
# Audio processing imports
from audiopro.audio.audio_loader import load_and_preprocess_audio
from audiopro.audio.extractor import extract_features
from audiopro.audio.metadata import get_file_metadata

# Output handling imports
from audiopro.output.output_handler import write_output
from audiopro.output.types import AudioAnalysis, FeatureConfig

# Utility imports
from audiopro.utils import optimized_convert_to_native_types, extract_rhythm
from audiopro.utils.logger import get_logger
from audiopro.utils.process import graceful_shutdown

# Monitoring imports
from audiopro.monitor.loader import load_monitor_functions

# Initialize logger
logger = get_logger(__name__)


async def analyze_audio(
    file_path: str,
    output_path: str,
    output_format: str = "msgpack",
    skip_monitoring: bool = False,
    feature_config: Optional[FeatureConfig] = None,
) -> None:
    """
    Analyze an audio file and extract features.

    Args:
        file_path: Path to the audio file to analyze
        output_path: Path where to save the analysis results (without extension)
        output_format: Format to save the results in ('msgpack' or 'json')
        skip_monitoring: Whether to skip performance monitoring
        feature_config: Optional configuration specifying which features to compute.
                      If None, all features will be computed.

    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If the audio file is invalid or too short
        RuntimeError: If the analysis fails critically
        Exception: If any other unexpected error occurs
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    start_time = time.time()

    # Initialize monitoring lists if monitoring is enabled
    cpu_usage_list = []
    active_cores_list = []
    monitoring_thread = None

    monitor_cpu_usage = None
    print_performance_stats = None

    if not skip_monitoring:
        # Only import monitoring functions if monitoring is enabled
        monitor_cpu_usage, print_performance_stats = load_monitor_functions()

    features = []

    def on_feature(feature):
        # Convert FrameFeatures to dict before appending
        features.append(feature.to_dict())

    with graceful_shutdown() as stop_flag:
        try:
            if not skip_monitoring and monitor_cpu_usage:
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
                    extract_features,
                    audio_data,
                    sample_rate,
                    on_feature,
                    feature_config,
                )

                logger.info("Analyzing rhythm patterns...")
                rhythm_future = executor.submit(extract_rhythm, audio_data)

                # Get results with logging
                logger.info("Waiting for task completion...")
                metadata = metadata_future.result()
                logger.info("Metadata extraction completed")

                features_future.result()  # Ensure all features are processed
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

            await write_output(analysis, output_path, output_format)

            end_time = time.time()
            if not skip_monitoring and print_performance_stats:
                print_performance_stats(
                    start_time, end_time, cpu_usage_list, active_cores_list
                )
            else:
                execution_time = end_time - start_time
                logger.info(
                    f"Execution Time: {execution_time:.4f} seconds ({execution_time*1000:.2f} ms)"
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
