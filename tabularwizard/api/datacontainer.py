import tabularmagic as tm
import pandas as pd
import logging
from pathlib import Path
from llama_index.experimental.query_engine import PandasQueryEngine
from .llms.openai import build_openai


logger_path = Path(__file__).parent / "_debug" / "_log.txt"
# clear the log file
with open(logger_path, "w") as f:
    f.write("")


class _DataContainer:
    def __init__(self):
        self.analyzer = None

    def set_analyzer(self, analyzer: tm.Analyzer):
        """Sets the Analyzer for the DataContainer."""
        self.analyzer = analyzer
        self.df = self.analyzer.datahandler().df_all()
        self.pd_query_engine = PandasQueryEngine(df=self.df, llm=build_openai())

    def update_df(self):
        """Update the DataFrame based on the Analyzer's state."""
        self.df = self.analyzer.datahandler().df_all()
        self.pd_query_engine = PandasQueryEngine(df=self.df, llm=build_openai())


GLOBAL_DATA_CONTAINER = _DataContainer()
"""Container for storing the Analyzer and DataFrame."""


def build_tabularmagic_analyzer(
    df: pd.DataFrame, df_test: pd.DataFrame | None = None, test_size: float = 0.2
) -> tm.Analyzer:
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

    tm.options.print_options.reset_logger(logger=tm_logger)
    analyzer = tm.Analyzer(df, df_test=df_test, test_size=test_size, verbose=True)
    return analyzer


def set_tabularmagic_analyzer(analyzer: tm.Analyzer) -> None:
    """Sets the TabularMagic Analyzer.

    Parameters
    ----------
    analyzer : tm.Analyzer
        The TabularMagic Analyzer.
    """
    GLOBAL_DATA_CONTAINER.set_analyzer(analyzer)
