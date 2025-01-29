import argparse


def parse_arguments():
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
