# typing imports
from typing import Dict, Any

# Standard library imports
import argparse

# Local application imports
from audiopro.output.types import (
    AVAILABLE_FEATURES,
    create_feature_config,
)
from audiopro.utils.path import SUPPORTED_FORMATS


def parse_arguments() -> Dict[str, Any]:
    """Parse command-line arguments for audio file processing.

    This function sets up and processes command-line arguments for the audio analysis tool.
    It handles input/output file paths, output format selection, monitoring options,
    and feature selection.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - input_file (str): Path to the input audio file to be analyzed
            - output_file (str): Name for the output file (without extension)
            - format (OutputFormat): Output format, either 'msgpack' (default) or 'json'
            - skip_monitoring (bool): Flag to disable performance monitoring if True
            - feature_config (Optional[FeatureConfig]): Configuration for which features to compute
            - time_range (Optional[TimeRange]): Time range for processing

    Example:
        args = parse_arguments()
        input_path = args.input_file
        output_name = args.output_file
        feature_config = args.feature_config
    """
    parser = argparse.ArgumentParser(
        description="Analyze audio files and extract features.",
        epilog="Note: Do not include file extensions in the output name. "
        "Use --format to specify the output format.",
    )

    parser.add_argument("input_file", type=str, help="Path to input audio file")

    parser.add_argument(
        "output_file",
        type=str,
        help="Name for output file WITHOUT extension. Use --format to specify format.",
    )

    parser.add_argument(
        "--format",
        type=str.lower,  # Convert to lowercase immediately
        choices=sorted(SUPPORTED_FORMATS),
        default="msgpack",
        help="Output format: 'msgpack' (default) or 'json'",
    )

    parser.add_argument(
        "--skip-monitoring",
        action="store_true",
        help="Skip performance monitoring to reduce overhead",
    )

    parser.add_argument(
        "--gzip",
        action="store_true",
        help="Enable gzip compression for msgpack output",
    )

    # Feature selection arguments
    feature_group = parser.add_argument_group("Feature Selection")
    feature_group.add_argument(
        "--features",
        type=str,
        nargs="+",
        choices=sorted(AVAILABLE_FEATURES),
        help="Select specific features to compute. If not specified, all features will be computed.",
    )

    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start time in seconds (default: 0.0)",
    )

    parser.add_argument(
        "--end",
        type=float,
        help="End time in seconds (default: entire file)",
    )

    args = parser.parse_args()

    # Create TimeRange dict if either start or end is specified
    time_range = {}
    if args.start > 0:
        time_range["start"] = args.start
    if args.end is not None:
        time_range["end"] = args.end

    # Convert args to dictionary with proper typing
    result = {
        "input_file": args.input_file,
        "output_file": args.output_file,
        "format": args.format,  # Already lowercase and validated as OutputFormat
        "skip_monitoring": args.skip_monitoring,
        "feature_config": create_feature_config(args.features),
        "time_range": time_range if time_range else None,
        "gzip_output": args.gzip,  # Pass gzip flag
    }

    return result
