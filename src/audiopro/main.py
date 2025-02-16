"""
This script serves as the entry point for the audio processing and performance monitoring tool.
It parses command-line arguments, sets up the output file, and runs the audio analysis asynchronously.

Modules:
    asyncio: Provides support for asynchronous programming.
    arg_parser: Local module for parsing command-line arguments.
    analyze_audio: Local module for performing audio analysis.

Functions:
    main: Parses command-line arguments and runs the audio analysis.

Usage:
    Run this script from the command line with the appropriate arguments to process an audio file.
Audio Processing and Performance Monitoring Tool
"""

# Standard library imports
import asyncio

# local CLI parsing
from .arg_parser import parse_arguments

# Analysis controller
from .analysis.controller import analyze_audio


if __name__ == "__main__":
    args = parse_arguments()

    asyncio.run(
        analyze_audio(
            args["input_file"],
            args["output_file"],
            output_format=args["format"],
            skip_monitoring=args["skip_monitoring"],
            feature_config=args["feature_config"],
            time_range=args["time_range"],
            gzip_output=args["gzip_output"],
            target_sample_rate=args["target_sample_rate"],  # Add new parameter
        )
    )
