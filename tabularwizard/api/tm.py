import tabularmagic as tm
import pandas as pd
import logging
from pathlib import Path


logger_path = Path(__file__).parent / "io" / "_tm_logger_output" / "log.txt"


tabularwizard_analyzer: tm.Analyzer = None


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
