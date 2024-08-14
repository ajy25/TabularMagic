import pandas as pd


class DataHandler_v2:
    """DataHandler: a class that handles all aspects of data
    preprocessing and loading."""

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        name: str | None = None,
        verbose: bool = True,
    ):
        """Initializes a DataHandler object.

        Parameters
        ----------
        df_train : pd.DataFrame
            The train DataFrame.

        df_test : pd.DataFrame
            The test DataFrame.

        name : str | None
            Default: None. The name of the DataHandler object.

        verbose : bool
            Default: True. If True, prints updates and warnings.
        """

        self._og_df_train = df_train
        self._og_df_test = df_test
        self._name = name
        self._verbose = verbose
