# typing imports
from typing import Union, Any

# third-party imports
import numpy as np


def optimized_convert_to_native_types(data: Any) -> Union[float, int, list, dict, Any]:
    """Convert numpy types to native Python types using vectorized operations"""
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
