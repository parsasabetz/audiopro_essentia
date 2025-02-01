# Standard imports
# Standard library imports
import argparse
from typing import Dict, Any

# Local application imports
from audiopro.output.types import FeatureConfig
from audiopro.utils.path import SUPPORTED_FORMATS, OutputFormat


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

    feature_set = [
        "rms",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_flatness",
        "spectral_rolloff",
        "zero_crossing_rate",
        "mfcc",
        "frequency_bands",
        "chroma",
    ]

    # Feature selection arguments
    feature_group = parser.add_argument_group("Feature Selection")
    feature_group.add_argument(
        "--features",
        type=str,
        nargs="+",
        choices=feature_set,
        help="Select specific features to compute. If not specified, all features will be computed.",
    )

    args = parser.parse_args()

    # Convert args to dictionary with proper typing
    result = {
        "input_file": args.input_file,
        "output_file": args.output_file,
        "format": args.format,  # Already lowercase and validated as OutputFormat
        "skip_monitoring": args.skip_monitoring,
    }

    # Create feature config if features were specified
    if args.features:
        feature_config: FeatureConfig = {}
        all_features = feature_set
        # Set selected features to True, others to False
        for feature in all_features:
            feature_config[feature] = feature in args.features
        result["feature_config"] = feature_config
    else:
        result["feature_config"] = None

    return result
