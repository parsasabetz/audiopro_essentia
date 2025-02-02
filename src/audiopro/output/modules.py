# Import necessary libraries for asynchronous file operations
import aiofiles

# Import necessary libraries for data serialization
import msgpack
import orjson

# Import custom types
from .types import AudioAnalysis


async def _write_json(data: AudioAnalysis, file_path: str) -> None:
    """
    Asynchronously write audio analysis data to a JSON file.

    Args:
        data (AudioAnalysis): The audio analysis data to be written to the file.
        file_path (str): The path to the file where the JSON data will be written.

    Returns:
        None

    Raises:
        OSError: If the file cannot be opened or written to.

    """
    json_bytes = orjson.dumps(
        data,
        option=orjson.OPT_SERIALIZE_NUMPY,
        default=lambda x: float(x) if hasattr(x, "dtype") else x,
    )

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(json_bytes)


async def _write_msgpack(data: AudioAnalysis, file_path: str) -> None:
    """
    Asynchronously writes the given audio analysis data to a file in MessagePack format.

    Args:
        data (AudioAnalysis): The audio analysis data to be written.
        file_path (str): The path to the file where the data will be written.

    Returns:
        None
    """
    packed_data = msgpack.packb(data, use_bin_type=True, strict_types=True)

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(packed_data)
