from functools import wraps
import pandas as pd
import numpy as np


def is_numerical(series: pd.Series | np.ndarray) -> bool:
    """Given a pandas Series or numpy array, check if it is numerical.

    Parameters
    ----------
    series : pd.Series | np.ndarray
        The series or array to check.

    Returns
    -------
    bool
        True if the series is numerical, False otherwise.
    """
    if isinstance(series, pd.Series):
        return pd.api.types.is_numeric_dtype(series)
    elif isinstance(series, np.ndarray):
        return np.issubdtype(series.dtype, np.number)
    else:
        raise ValueError("Input must be a pandas Series or numpy array.")


def check_list_uniqueness(lst: list):
    """
    Check if all elements in a list are unique.

    Parameters
    ----------
    lst : list
        The list to check.

    Raises
    ------
    ValueError
        If any element is duplicated.
    """
    seen = set()
    for index, item in enumerate(lst):
        if item in seen:
            raise ValueError(f"Duplicate element '{item}' found at index {index}.")
        seen.add(item)


def ensure_arg_list_uniqueness(*list_arg_names):
    """
    Decorator that checks if all elements in a list argument are unique.

    Parameters
    ----------
    *list_arg_names : str
        The names of the list arguments to check.

    Raises
    ------
    ValueError
        If any element is duplicated.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check each argument
            for arg_name, arg_value in kwargs.items():
                if isinstance(arg_value, list):
                    seen = set()
                    for index, item in enumerate(arg_value):
                        if item in seen:
                            raise ValueError(
                                f"Duplicate element '{item}' found at index {index} "
                                f"in list argument '{arg_name}'"
                            )
                        seen.add(item)

            # Also check positional arguments
            for i, arg in enumerate(args):
                if isinstance(arg, list):
                    seen = set()
                    for index, item in enumerate(arg):
                        if item in seen:
                            raise ValueError(
                                f"Duplicate element '{item}' found at index {index} "
                                f"in positional list argument {i}"
                            )
                        seen.add(item)

            # If all checks pass, call the original function
            return func(*args, **kwargs)

        return wrapper

    return decorator
