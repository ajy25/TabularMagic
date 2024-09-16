import pandas as pd
from pathlib import Path


path_to_csv_cache = Path(__file__).parent / "_csv_cache"
path_to_csv_cache.mkdir(exist_ok=True)


class _CSVCacheTracker:
    """Tracks the cache of CSV files that have been created. Prevents
    overflow of the cache by deleting the oldest files.
    """

    def __init__(self, max_files: int):
        """Initializes the CSV cache tracker.

        Parameters
        ----------
        max_files : int
            The maximum number of files to store in the cache.
        """
        self.max_files = max_files
        self.num_to_filepath = {
            i: path_to_csv_cache / f"file_{i}.csv" for i in range(max_files)
        }
        self.oldest_to_youngest = list(range(max_files))

    def save_df_to_csv(self, df: pd.DataFrame, save_index: bool = True):
        """Saves a DataFrame to a CSV file and updates the cache tracker.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to save.

        save_index : bool, optional
            Whether to save the index of the DataFrame.
        """
        oldest_file = self.num_to_filepath[self.oldest_to_youngest[0]]
        df.to_csv(oldest_file, index=save_index)
        self.oldest_to_youngest.append(self.oldest_to_youngest.pop(0))


csv_cache_tracker = _CSVCacheTracker(5)


def save_df_to_csv(df: pd.DataFrame, save_index: bool = True):
    """Saves a DataFrame to a CSV file and updates the cache tracker.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.

    save_index : bool, optional
        Whether to save the index of the DataFrame.
    """
    csv_cache_tracker.save_df_to_csv(df, save_index)
