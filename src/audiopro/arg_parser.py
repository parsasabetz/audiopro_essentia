# Standard imports
import argparse


def parse_arguments():
    """Parse command-line arguments for audio file processing.

    This function sets up and processes command-line arguments for the audio analysis tool.
    It handles input/output file paths, output format selection, and monitoring options.

    Returns:
        argparse.Namespace: An object containing the following attributes:
            - input_file (str): Path to the input audio file to be analyzed
            - output_file (str): Name for the output file (without extension)
            - format (str): Output format, either 'msgpack' (default) or 'json'
            - skip_monitoring (bool): Flag to disable performance monitoring if True

    Example:
        args = parse_arguments()
        input_path = args.input_file
        output_name = args.output_file
    """
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
    return parser.parse_args()
