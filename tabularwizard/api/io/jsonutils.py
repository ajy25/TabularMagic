import json
import pandas as pd
from pathlib import Path
from typing import Any


path_to_json_cache = Path(__file__).parent / "_json_cache"
path_to_json_cache.mkdir(exist_ok=True)


class _JSONCacheTracker:
    """Tracks the cache of JSON files that have been created. Prevents
    overflow of the cache by deleting the oldest files.
    """

    def __init__(self, max_files: int):
        """Initializes the JSON cache tracker.

        Parameters
        ----------
        max_files : int
            The maximum number of files to store in the cache.
        """
        self.max_files = max_files
        self.num_to_filepath = {
            i: path_to_json_cache / f"file_{i}.json" for i in range(max_files)
        }
        self.oldest_to_youngest = list(range(max_files))

    def save_to_json(self, data: Any) -> str:
        """Saves data to a JSON file and updates the cache tracker.
        Returns a JSON string containing the data.

        Parameters
        ----------
        data : Any
            The data to save (must be JSON serializable).

        Returns
        -------
        str
            The JSON string containing the data.
        """
        oldest_file = self.num_to_filepath[self.oldest_to_youngest[0]]
        with open(oldest_file, "w") as f:
            json.dump(data, f, indent=2)
        self.oldest_to_youngest.append(self.oldest_to_youngest.pop(0))
        return json.dumps(data, indent=2)

    def read_json(self, name: str) -> str:
        """Reads a JSON file and returns its contents as a JSON-formatted string.
        The name must be in the form of 'file_{integer}.json'.
        For example, 'file_1.json'.

        Parameters
        ----------
        name : str
            The name of the JSON file to read.

        Returns
        -------
        str
            The JSON string containing the data.
        """
        try:
            file_number = int(name.split("_")[1].split(".")[0])
            file_path = self.num_to_filepath[file_number]
            with open(file_path, "r") as f:
                data = json.load(f)
                return json.dumps(data)
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid file name format: {name}") from e
        except KeyError:
            raise KeyError(f"File number {file_number} not found in the mapping.")
        except FileNotFoundError:
            raise FileNotFoundError(f"File {file_path} not found.")
        except json.JSONDecodeError:
            raise ValueError(f"File {file_path} contains invalid JSON.")


json_cache_tracker = _JSONCacheTracker(5)


def save_df_to_json(df: pd.DataFrame, save_index: bool = True) -> str:
    """Saves a DataFrame to a JSON file and updates the cache tracker.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.

    save_index : bool
        Whether to save the index of the DataFrame. Defaults to True.

    Returns
    -------
    str
        The JSON string containing the DataFrame.
    """
    if save_index:
        json_data = df.to_dict(orient="index", index=True)
    else:
        json_data = df.to_dict(orient="records")
    return json_cache_tracker.save_to_json(json_data)


def save_dict_to_json(data: dict) -> str:
    """Saves a dictionary to a JSON file and updates the cache tracker.

    Parameters
    ----------
    data : dict
        The dictionary to save.

    Returns
    -------
    str
        The JSON string containing the dictionary.
    """
    return json_cache_tracker.save_to_json(data)
