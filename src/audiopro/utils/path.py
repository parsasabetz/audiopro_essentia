"""Utilities for path handling and validation."""

# typing imports
from typing import Literal

# standard imports
from pathlib import Path

# Define supported output formats
OutputFormat = Literal["json", "msgpack"]
SUPPORTED_FORMATS: set[str] = {"json", "msgpack"}
FORMAT_EXTENSIONS = {
    "json": ".json",
    "msgpack": ".msgpack"
}


def validate_and_process_output_path(output_path: str, output_format: OutputFormat) -> str:
    """
    Validate output path and ensure it doesn't contain an extension.
    Append the appropriate extension based on the format.

    Args:
        output_path: The output path specified by the user (should not have extension)
        output_format: The output format, already validated and normalized by argparse

    Returns:
        str: The processed output path with the correct extension

    Raises:
        ValueError: If the output path contains an extension
    """
    path = Path(output_path)
    
    # Check if path has an extension
    if path.suffix:
        raise ValueError(
            f"Output path should not include an extension. "
            f"Got '{output_path}'. Please provide only the name without extension "
            f"and use --format to specify the output format (json or msgpack)."
        )

    # Add the appropriate extension based on format (already validated)
    return str(path) + FORMAT_EXTENSIONS[output_format]
