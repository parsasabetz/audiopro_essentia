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

# Moved extractor functions to a separate module
from extractor import extract_features

# Moved monitoring functions to a separate module
from monitor import monitor_cpu_usage, print_performance_stats

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

def analyze_audio(file_path: str, output_file: str) -> None:
    """
    Main function to analyze audio and monitor performance.
    
    Args:
        file_path: Path to input audio file
        output_file: Path for output JSON file
    """
    start_time = time.time()
    cpu_usage_list: List[float] = []
    active_cores_list: List[int] = []
    stop_flag = threading.Event()
    
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
        
        # Save to JSON
        with open(output_file, "w") as f:
            json.dump(analysis, f, indent=4)
            
        logger.info(f"Analysis saved to {output_file}")
        
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
    parser.add_argument("output_file", type=str, help="Path to output JSON file")
    args = parser.parse_args()
    analyze_audio(args.input_file, args.output_file)