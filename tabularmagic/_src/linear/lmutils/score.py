import pandas as pd
from typing import Literal
import numpy as np
import statsmodels.api as sm


def score_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    feature_list: list[str],
    model: Literal["ols", "binomial", "poisson", "negbinomial"],
    metric: Literal["aic", "bic"],
) -> float:
    """Scores a linear model.

    Parameters
    ----------
    X_train : pd.DataFrame
        The training data.

    y_train : pd.DataFrame
        The target data.

    feature_list : list[str]
        The list of features to use in the model.

    model : Literal["ols", "binomial", "poisson", "negbinomial"]
        The model to use.

    metric : Literal["aic", "bic"]
        The metric to use for scoring.
    """
    if len(feature_list) == 0:
        return np.inf

    subset_X_train = X_train[feature_list]
    subset_X_train_with_constant = sm.add_constant(subset_X_train, has_constant="add")

    if model == "ols":
        new_model = sm.OLS(y_train, subset_X_train_with_constant)
    elif model == "binomial":
        new_model = sm.GLM(
            y_train, subset_X_train_with_constant, family=sm.families.Binomial()
        )
    elif model == "poisson":
        new_model = sm.GLM(
            y_train, subset_X_train_with_constant, family=sm.families.Poisson()
        )
    elif model == "negbinomial":
        new_model = sm.GLM(
            y_train, subset_X_train_with_constant, family=sm.families.NegativeBinomial()
        )
    else:
        raise ValueError(
            "Model must be one of 'ols', 'binomial', 'poisson', 'negbinomial'"
        )

    if metric == "aic":
        return new_model.fit().aic
    elif metric == "bic":
        return new_model.fit().bic
    else:
        raise ValueError("Metric must be one of 'aic', 'bic'")
