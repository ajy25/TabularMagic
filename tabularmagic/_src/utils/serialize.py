import numpy as np
import pandas as pd


def prepare_for_json(data):
    """
    Convert a dictionary containing pandas/numpy data types into JSON-serializable format.
    Handles MultiIndex by converting tuple keys to strings.

    Args:
        data: Dictionary, DataFrame, or other data structure to convert

    Returns:
        JSON-serializable version of the data
    """
    if isinstance(data, dict):
        return {
            # Convert tuple keys to string representation
            str(k) if isinstance(k, tuple) else str(k): prepare_for_json(v)
            for k, v in data.items()
        }
    elif isinstance(data, (list, tuple)):
        return [prepare_for_json(x) for x in data]
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        if isinstance(data.index, pd.MultiIndex):
            # For MultiIndex, convert to dict with string keys
            return {
                str(idx): prepare_for_json(row.to_dict())
                for idx, row in data.iterrows()
            }
        return prepare_for_json(data.to_dict())
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return prepare_for_json(data.tolist())
    elif pd.isna(data):
        return None
    return data
