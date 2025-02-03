# typing imports
from typing import Union, Any

# third-party imports
import numpy as np


def optimized_convert_to_native_types(data: Any) -> Union[float, int, list, dict, Any]:
    """Convert complex data types to native Python types.

    This function recursively converts NumPy types, arrays and nested structures
    into their Python native equivalents.

    Args:
        data (Any): Input data to be converted. Can be:
            - NumPy scalar types (float, int, bool)
            - NumPy arrays
            - Dictionaries (potentially nested)
            - Lists (potentially nested)
            - Native Python types (which will be returned as-is)

    Returns:
        Union[float, int, list, dict, Any]: Converted data in native Python format:
            - NumPy floating types -> float
            - NumPy integer types -> int
            - NumPy boolean types -> bool
            - NumPy arrays -> list
            - Dictionaries -> dict with converted values
            - Lists -> list with converted values
            - Other types are returned unchanged

    Examples:
        >>> import numpy as np
        >>> optimized_convert_to_native_types(np.float32(1.5))
        1.5
        >>> optimized_convert_to_native_types(np.array([1, 2, 3]))
        [1, 2, 3]
        >>> optimized_convert_to_native_types({'a': np.int64(1), 'b': [np.float32(2.5)]})
        {'a': 1, 'b': [2.5]}
    """
    if data is None or isinstance(data, (bool, int, float, str)):
        return data

    # Use np.issubdtype to capture any numpy scalar
    if isinstance(data, np.generic):
        if np.issubdtype(data.dtype, np.floating):
            return float(data)
        if np.issubdtype(data.dtype, np.integer):
            return int(data)
        if np.issubdtype(data.dtype, np.bool_):
            return bool(data)

    if isinstance(data, np.ndarray):
        return data.tolist()

    if isinstance(data, dict):
        return {k: optimized_convert_to_native_types(v) for k, v in data.items()}
    if isinstance(data, list):
        return [optimized_convert_to_native_types(x) for x in data]

    return data
