import pandas as pd
from .._src import (
    GLOBAL_DATA_CONTAINER,
    GLOBAL_IO,
    build_tabularmagic_analyzer,
    set_tabularmagic_analyzer,
    print_debug,
)

from .._src.agents.eda_agent import build_eda_agent


class TabularWizard:
    """Class for interacting with the LLMs for data analysis on tabular data."""

    def __init__(
        self,
        df: pd.DataFrame,
        df_test: pd.DataFrame | None = None,
        test_size: float = 0.2,
    ):
        """Initializes the TabularWizard object.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to build the Analyzer for.

        df_test : pd.DataFrame | None
            The test DataFrame to use for the Analyzer. Defaults to None.

        test_size : float
            The size of the test set. Defaults to 0.2.
        """
        set_tabularmagic_analyzer(
            build_tabularmagic_analyzer(df, df_test=df_test, test_size=test_size)
        )

        print_debug(
            "TabularWizard initialized with Analyzer built from the provided DataFrame."
        )

        self.data_container = GLOBAL_DATA_CONTAINER

        print_debug(
            "Data container initialized with the Analyzer built from the "
            "provided DataFrame."
        )

        self.io = GLOBAL_IO

        print_debug("Global IO initialized.")

        self._eda_agent = build_eda_agent()

    def chat(message: str) -> str:
        """Interacts with the LLM to provide data analysis insights.

        Parameters
        ----------
        message : str
            The message to send to the LLM.

        Returns
        -------
        str
            The response from the LLM.
        """
        raise NotImplementedError("This method is not implemented yet.")
