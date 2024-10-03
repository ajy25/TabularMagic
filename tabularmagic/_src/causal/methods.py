import numpy as np
import pandas as pd
from typing import Literal

from .utils.models import _SimpleLogit, _SimpleOLS


def estimate_ate(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: list[str],
    method: Literal["naive", "simple_ols"],
) -> tuple[float, float]:
    """Estimates the average treatment effect (ATE) using the given method.

    Parameters
    ----------
    data : pd.DataFrame
        The data for which to estimate the ATE.

    treatment : str
        The name of the treatment variable.

    outcome : str
        The name of the outcome variable.

    confounders : list[str]
        The names of the confounding variables.

    method : str
        The method to use for estimating the ATE.

        - "naive":
            Estimates the ATE by computing the difference in means between the
            treatment and control groups. Does not account for confounding.

        - "simple_ols":
            Estimates the ATE using simple linear regression. Assumes no
            interaction between the treatment variable and the confounders.

    Returns
    -------
    tuple[float, float]
        The estimated ATE and its standard error.
    """
    if type(confounders) != list:
        raise ValueError("Confounders must be a list of strings.")
    else:
        for confounder in confounders:
            if confounder not in data.columns:
                raise ValueError(f"Confounder {confounder} not found in data.")

    if type(treatment) != str:
        raise ValueError("Treatment must be a string.")
    elif treatment not in data.columns:
        raise ValueError(f"Treatment {treatment} not found in data.")

    if type(outcome) != str:
        raise ValueError("Outcome must be a string.")
    elif outcome not in data.columns:
        raise ValueError(f"Outcome {outcome} not found in data.")

    if method == "naive":
        # Estimates the ATE by computing the difference in means between the
        # treatment and control groups. Does not account for confounding.
        ate = data.groupby(treatment)[outcome].mean().diff().iloc[-1]
        se = np.sqrt(
            data.groupby(treatment)[outcome].var().sum()
            / data.groupby(treatment)[outcome].count().sum()
        )
        return ate, se

    elif method == "simple_ols":
        # Fits an OLS model with the treatment variable and confounders.
        # Interprets the coefficient of the treatment variable as the ATE.
        model = _SimpleOLS(
            data=data, target=outcome, predictors=[treatment] + confounders
        )
        ate = model.get_coef(treatment)
        se = model.get_se(treatment)
        return ate, se

    else:
        raise ValueError(f"Method {method} is not supported.")
