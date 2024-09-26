import pandas as pd
from typing import Literal
from sklearn.impute._base import _BaseImputer
from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer


def impute(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    vars: list[str],
    numeric_strategy: Literal["median", "mean", "5nn"],
    categorical_strategy: Literal["most_frequent", "constant"],
):
    """Imputes missing values for the subsets of the train and test datasets.
    Methods are fitted on the train dataset and applied to both datasets.

    Parameters
    ----------
    df_train : pd.DataFrame
        Train dataset.

    df_test : pd.DataFrame
        Test dataset.

    vars : list[str]
        Variables to impute.

    numeric_strategy : Literal["median", "mean", "5nn"]
        Strategy to impute numerical variables.

    categorical_strategy : Literal["most_frequent", "constant"]
        Strategy to impute categorical variables.

    Returns
    -------
    pd.DataFrame
        Train dataset with imputed values.

    pd.DataFrame
        Test dataset with imputed values.

    _BaseImputer
        The imputer for numeric variables.

    _BaseImputer
        The imputer for categorical variables.
    """
    

