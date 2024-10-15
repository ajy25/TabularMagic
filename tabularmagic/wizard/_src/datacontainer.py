from ... import Analyzer
from ...options import print_options

import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine

import logging


from .llms.openai import build_openai
from ._debug.logger import logger_path


GLOBAL_DATA_CONTAINER = None
build_tabularmagic_analyzer = None
set_tabularmagic_analyzer = None


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
        self.pd_query_engine = PandasQueryEngine(df=self.df, llm=build_openai())

    def update_df(self):
        """Update the DataFrame based on the Analyzer's state."""
        self.df = self.analyzer.datahandler().df_all()
        self.variable_info = VariableInfo(vars=self.df.columns.to_list())
        self.pd_query_engine = PandasQueryEngine(df=self.df, llm=build_openai())


GLOBAL_DATA_CONTAINER = DataContainer()
"""Container for storing the Analyzer and DataFrame."""


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
    tm_logger = logging.Logger(name="tmwizard_magic_logger", level=logging.INFO)
    filehandler = logging.FileHandler(logger_path)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    filehandler.setFormatter(formatter)
    tm_logger.addHandler(filehandler)

    print_options.reset_logger(logger=tm_logger)
    analyzer = Analyzer(df, df_test=df_test, test_size=test_size, verbose=True)
    return analyzer


def set_tabularmagic_analyzer(analyzer: Analyzer) -> None:
    """Sets the TabularMagic Analyzer.

    Parameters
    ----------
    analyzer : tm.Analyzer
        The TabularMagic Analyzer.
    """
    GLOBAL_DATA_CONTAINER.set_analyzer(analyzer)
