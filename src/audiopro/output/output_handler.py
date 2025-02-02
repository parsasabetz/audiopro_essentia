"""
Handles writing the analysis results to a file in either JSON or MessagePack format.
"""

# Standard library imports
from typing import Callable, Dict, Any

# Local imports
from .types import AudioAnalysis
from ..utils.logger import get_logger
from ..utils.path import validate_and_process_output_path, OutputFormat
from .modules import _write_json, _write_msgpack

# Configure logging
logger = get_logger(__name__)

# Map output formats to their handlers
OUTPUT_HANDLERS: Dict[str, Callable[[AudioAnalysis, str], Any]] = {
    "json": _write_json,
    "msgpack": _write_msgpack,
}


async def write_output(
    analysis: AudioAnalysis, output_path: str, output_format: OutputFormat
):
    """
    Asynchronously writes analysis results to a file in either JSON or MessagePack format.

    Args:
        analysis (AudioAnalysis): Dictionary containing the analysis results to be written
        output_path (str): Path to the output file (without extension)
        output_format (OutputFormat): Format to save the file in, already validated by argparse

    Raises:
        aiofiles.errors.FileError: If there are issues with file operations
        orjson.JSONEncodeError: If JSON encoding fails
        msgpack.exceptions.PackException: If MessagePack encoding fails
        Exception: For any other exceptions that occur during file operations

    Note:
        The analysis object is expected to be already filtered for null values:
        - Features are only created if requested
        - FrameFeatures.to_dict() excludes None values
        - Other fields are required by type definitions
    """
    # Process and validate the output path, adding the correct extension
    final_output = validate_and_process_output_path(output_path, output_format)

    logger.info(f"Writing output to {final_output} in {output_format} format...")

    try:
        await OUTPUT_HANDLERS[output_format](analysis, final_output)
        logger.info("Output written successfully")
    except Exception as e:
        logger.error("Failed to write output: %s", str(e))
        raise
