# typing imports
from typing import Union, Any

# third-party imports
import numpy as np


def optimized_convert_to_native_types(data: Any) -> Union[float, int, list, dict, Any]:
    """Convert numpy types to native Python types using vectorized operations"""
    # Fast path for most common types
    if data is None or isinstance(data, (bool, int, float, str)):
        return data

    # Numpy scalar types (most common after basic types)
    if isinstance(data, (np.float32, np.float64)):
        return float(data)
    if isinstance(data, (np.int32, np.int64)):
        return int(data)
    if isinstance(data, np.bool_):
        return bool(data)

    # Numpy arrays (handled in one go)
    if isinstance(data, np.ndarray):
        return data.tolist()

    # Collections (using generators for memory efficiency)
    if isinstance(data, dict):
        return {k: optimized_convert_to_native_types(v) for k, v in data.items()}
    if isinstance(data, list):
        return [optimized_convert_to_native_types(x) for x in data]

    return data
