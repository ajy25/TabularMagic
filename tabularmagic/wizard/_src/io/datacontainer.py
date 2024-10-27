import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
import logging

from ..options import options
from .._debug.logger import debug_log_path
from .... import Analyzer
from ....options import print_options


def build_tabularmagic_analyzer(
    df: pd.DataFrame, df_test: pd.DataFrame | None = None, test_size: float = 0.2
) -> Analyzer:
    """Builds a TabularMagic Analyzer for a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to build the Analyzer for.

    df_test : pd.DataFrame | None
        The test DataFrame to use for the Analyzer. Defaults to None.

    test_size : float
        The size of the test set. Defaults to 0.2.

    Returns
    -------
    tm.Analyzer
        The TabularMagic Analyzer.
    """
    print_options.reset_logger(logging.Logger("Blank"))
    debug_logger = logging.Logger("Analyzer Debug Log")
    filehandler = logging.FileHandler(debug_log_path)
    style = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    filehandler.setFormatter(style)
    debug_logger.addHandler(filehandler)
    print_options.reset_secondary_logger(logger=debug_logger)

    analyzer = Analyzer(df, df_test=df_test, test_size=test_size, verbose=True)
    return analyzer


class VariableInfo:
    def __init__(
        self,
        vars: list[str],
    ):
        """Initializes the VariableInfo object.

        Parameters
        ----------
        vars : list[str]
            The variables to provide information on.
        """
        self.vars_to_description = {var: None for var in vars}

    def set_description(
        self,
        var: str,
        description: str,
    ) -> None:
        """Sets the description for a variable.

        Parameters
        ----------
        var : str
            The variable to set the description for.

        description : str
            The description of the variable.
        """
        self.vars_to_description[var] = description

    def get_description(
        self,
        var: str,
    ) -> str:
        """Gets the description for a variable.

        Parameters
        ----------
        var : str
            The variable to get the description for.

        Returns
        -------
        str
            The description of the variable.
        """
        output = self.vars_to_description[var]
        if output is None:
            return ""


class DataContainer:
    def __init__(self):
        self.analyzer = None

    def set_analyzer(self, analyzer: Analyzer):
        """Sets the Analyzer for the DataContainer."""
        self.analyzer = analyzer
        self.df = self.analyzer.datahandler().df_all()
        self.variable_info = VariableInfo(vars=self.df.columns.to_list())
        self.pd_query_engine = PandasQueryEngine(
            df=self.df, llm=options.llm_build_function()
        )

    def update_df(self):
        """Update the DataFrame based on the Analyzer's state."""
        self.df = self.analyzer.datahandler().df_all()
        self.variable_info = VariableInfo(vars=self.df.columns.to_list())
        self.pd_query_engine = PandasQueryEngine(
            df=self.df, llm=options.llm_build_function()
        )
