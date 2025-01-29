import numpy as np


def optimized_convert_to_native_types(data):
    """Convert numpy types to native Python types using vectorized operations"""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)
    elif isinstance(data, dict):
        return {
            key: optimized_convert_to_native_types(value) for key, value in data.items()
        }
    elif isinstance(data, list):
        return list(map(optimized_convert_to_native_types, data))
    return data
