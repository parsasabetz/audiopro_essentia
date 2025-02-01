"""
Handles writing the analysis results to a file in either JSON or MessagePack format.
"""

# Third-party imports
import aiofiles
import msgpack
import orjson

# Local imports
from .types import AudioAnalysis
from ..utils.logger import get_logger
from ..utils.path import validate_and_process_output_path

# Configure logging
logger = get_logger(__name__)


async def write_output(analysis: AudioAnalysis, output_path: str, output_format: str):
    """
    Asynchronously writes analysis results to a file in either JSON or MessagePack format.

    Args:
        analysis (AudioAnalysis): Dictionary containing the analysis results to be written
        output_path (str): Path to the output file (without extension)
        output_format (str): Format to save the file in - either "json" or "msgpack"

    Raises:
        aiofiles.errors.FileError: If there are issues with file operations
        orjson.JSONEncodeError: If JSON encoding fails
        msgpack.exceptions.PackException: If MessagePack encoding fails
        ValueError: If the output_format is not 'json' or 'msgpack'
        Exception: For any other exceptions that occur during file operations

    Note:
        The analysis object is expected to be already filtered for null values:
        - Features are only created if requested
        - FrameFeatures.to_dict() excludes None values
        - Other fields are required by type definitions
    """
    # Process and validate the output path, adding the correct extension
    final_output = validate_and_process_output_path(output_path, output_format)

    logger.info("Output format: %s", output_format)
    logger.info("Output file will be: %s", final_output)
    logger.info("Writing output to %s...", final_output)

    try:
        # No need to check output_format.lower() since it's already validated
        if output_format == "json":
            # Convert to JSON with orjson's default handler for numpy types
            json_bytes = orjson.dumps(
                analysis,
                option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2,
                default=lambda x: float(x) if hasattr(x, "dtype") else x,
            )
            async with aiofiles.open(final_output, "wb") as f:
                await f.write(json_bytes)
        else:  # Must be msgpack since format is validated
            # Use buffer for large MessagePack data
            packed_data = msgpack.packb(analysis, use_bin_type=True)
            async with aiofiles.open(final_output, "wb") as f:
                await f.write(packed_data)

        logger.info("Output written successfully")
    except Exception as e:
        logger.error("Failed to write output: %s", str(e))
        raise
