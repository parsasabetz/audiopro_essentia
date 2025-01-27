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

# Local imports - Changed from absolute to relative imports
from .extractor import extract_features
from .monitor import monitor_cpu_usage, print_performance_stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
FRAME_LENGTH = 2048
HOP_LENGTH = 512
FREQUENCY_BANDS = {
    "low": 250,     # Hz
    "mid": 2000,    # Hz
    # high: everything above mid
}

def analyze_audio(file_path: str, output_file: str, output_format: str = "json") -> None:
    """
    Main function to analyze audio and monitor performance.
    
    Args:
        file_path: Path to input audio file
        output_file: Path for output file (without extension)
        output_format: Format of the output file ('json' or 'msgpack')
    """
    # Input validation
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")
    if not isinstance(output_file, str):
        raise TypeError("output_file must be a string")
    if not isinstance(output_format, str):
        raise TypeError("output_format must be a string")
    
    # Normalize output format
    output_format = output_format.lower().strip()
    if output_format not in ["json", "msgpack"]:
        raise ValueError("output_format must be either 'json' or 'msgpack'")
    
    # Remove any existing extension from output_file
    output_file = os.path.splitext(output_file)[0]
    
    # Add the correct extension
    final_output = f"{output_file}.{output_format}"
    
    logger.info(f"Output format: {output_format}")
    logger.info(f"Output file will be: {final_output}")
    
    start_time = time.time()
    cpu_usage_list: List[float] = []
    active_cores_list: List[int] = []
    stop_flag = threading.Event()
    
    # Validate input parameters
    if not isinstance(output_format, str) or output_format.lower() not in ["json", "msgpack"]:
        raise ValueError("output_format must be either 'json' or 'msgpack'")
    
    # Remove any existing extension from output_file
    output_file = os.path.splitext(output_file)[0]
    
    # Add the correct extension based on format
    output_format = output_format.lower()
    output_path = f"{output_file}.{output_format}"
    
    # Start performance monitoring
    monitoring_thread = threading.Thread(
        target=monitor_cpu_usage,
        args=(psutil.Process(os.getpid()), cpu_usage_list, active_cores_list, stop_flag)
    )
    monitoring_thread.start()
    
    try:
        logger.info(f"Loading audio file: {file_path}")
        audio_data, sample_rate = librosa.load(file_path)
        
        # Get tempo and beats with proper type conversion
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
        # Fix: Replace np.asscalar with item()
        tempo = float(tempo.item() if isinstance(tempo, np.ndarray) else tempo)
        beat_times = [float(t) for t in librosa.frames_to_time(beats, sr=sample_rate)]
        
        logger.info("Extracting features...")
        features = extract_features(audio_data, sample_rate)
        
        # Create analysis dictionary with proper types
        analysis = {
            "tempo": tempo,
            "beats": beat_times,
            "features": features
        }
        
        # Save to file based on the specified format
        if output_format == "json":
            with open(final_output, "w") as f:
                json.dump(analysis, f, indent=4)
            logger.info(f"Analysis saved to {final_output}")
        elif output_format == "msgpack":
            with open(final_output, "wb") as f:
                msgpack.pack(analysis, f)
            logger.info(f"Analysis saved to {final_output}")
        
        # Stop CPU monitoring
        stop_flag.set()
        monitoring_thread.join()
        
        end_time = time.time()
        print_performance_stats(start_time, end_time, cpu_usage_list, active_cores_list)
        
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
    parser.add_argument("output_file", type=str, help="Name for output file without extension")
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "msgpack"],
        default="json",
        help="Output format: 'json' (default) or 'msgpack'"
    )
    args = parser.parse_args()
    
    # Validate and append the appropriate file extension
    if args.output_file.endswith(('.json', '.msgpack')):
        raise ValueError("Please provide the output file name without an extension.")
    
    if args.format == "json":
        output_file = f"{args.output_file}.json"
    else:
        output_file = f"{args.output_file}.msgpack"
    
    analyze_audio(args.input_file, output_file, output_format=args.format)