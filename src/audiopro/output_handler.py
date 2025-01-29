# Standard library imports
import logging

# Third-party imports
import aiofiles
import msgpack
import orjson

# Configure logging
logger = logging.getLogger(__name__)


async def write_output(analysis: dict, final_output: str, output_format: str):
    """
    Asynchronously writes analysis results to a file in either JSON or MessagePack format.

    Args:
        analysis (dict): Dictionary containing the analysis results to be written
        final_output (str): Path to the output file where results will be saved
        output_format (str): Format to save the file in - either "json" or MessagePack

    Returns:
        None

    Raises:
        aiofiles.errors.FileError: If there are issues with file operations
        orjson.JSONEncodeError: If JSON encoding fails
        msgpack.exceptions.PackException: If MessagePack encoding fails

    Note:
        The function uses orjson for JSON encoding and msgpack for MessagePack format.
        File operations are performed asynchronously using aiofiles.
    """
    # Use asynchronous file writing
    if output_format == "json":
        async with aiofiles.open(final_output, "w") as f:
            await f.write(orjson.dumps(analysis).decode())
    else:
        # Serialize the analysis dictionary to MessagePack bytes
        packed_data = msgpack.packb(analysis)
        async with aiofiles.open(final_output, "wb") as f:
            await f.write(packed_data)

    logger.info(f"Analysis saved to {final_output}")
