import pandas as pd
from .tabularmagic_utils import build_tabularmagic_analyzer, set_tabularmagic_analyzer


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
        self.analyzer = build_tabularmagic_analyzer(
            df, df_test=df_test, test_size=test_size
        )
        set_tabularmagic_analyzer(self.analyzer)

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
