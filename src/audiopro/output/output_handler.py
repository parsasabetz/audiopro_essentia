# Standard library imports
# Third-party imports
import aiofiles
import msgpack
import orjson

# Local imports
from .types import AudioAnalysis
from ..utils.logger import get_logger

# Configure logging
logger = get_logger(__name__)


async def write_output(analysis: AudioAnalysis, final_output: str, output_format: str):
    """
    Asynchronously writes analysis results to a file in either JSON or MessagePack format.

    Args:
        analysis (AudioAnalysis): Dictionary containing the analysis results to be written
        final_output (str): Path to the output file where results will be saved
        output_format (str): Format to save the file in - either "json" or MessagePack

    Raises:
        aiofiles.errors.FileError: If there are issues with file operations
        orjson.JSONEncodeError: If JSON encoding fails
        msgpack.exceptions.PackException: If MessagePack encoding fails
        ValueError: If the output_format is not 'json' or 'msgpack'

    Note:
        The function uses orjson for JSON encoding and msgpack for MessagePack format.
        File operations are performed asynchronously using aiofiles.
    """
    output_format = output_format.lower().strip()
    if output_format not in ["json", "msgpack"]:
        raise ValueError("output_format must be either 'json' or 'msgpack'")

    logger.info("Output format: %s", output_format)
    logger.info("Output file will be: %s", final_output)
    logger.info("Writing output to %s...", final_output)

    # Use asynchronous file writing with proper error handling
    try:
        if output_format == "json":
            async with aiofiles.open(final_output, "w") as f:
                await f.write(orjson.dumps(analysis).decode())
        else:
            # Serialize the analysis dictionary to MessagePack bytes
            packed_data = msgpack.packb(analysis)
            async with aiofiles.open(final_output, "wb") as f:
                await f.write(packed_data)

        logger.info("Output written successfully")
    except (
        aiofiles.errors.FileError,
        orjson.JSONEncodeError,
        msgpack.exceptions.PackException,
    ) as e:
        logger.error("Failed to write output: %s", str(e))
        raise
