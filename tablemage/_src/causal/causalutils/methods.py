import pandas as pd
from typing import Literal
from ...._src.ml.predict.classification import BaseC


def compute_weights_from_propensity_scores(
    estimand: Literal["ate", "att"],
    propensity_scores: pd.Series,
    treatment: pd.Series,
) -> pd.Series:
    """Computes the weights for the inverse propensity weighting (IPW) estimator.

    Parameters
    ----------
    estimand : Literal["ate", "att"]
        The estimand of interest. 'ate' for Average Treatment Effect,
        'att' for Average Treatment on the Treated.

    propensity_scores : pd.Series
        The propensity scores for the observations.

    treatment : pd.Series
        The treatment indicator variable.

    Returns
    -------
    pd.Series
        The weights for the observations.
    """
    # ensure indices for propensity_scores and treatment are the same
    if not propensity_scores.index.equals(treatment.index):
        raise ValueError(
            "Indices for propensity_scores and treatment must be the same."
        )

    # initialize output series as zeros
    output = pd.Series(0.0, index=propensity_scores.index)

    # Boolean masks for treated and control groups
    idx_for_treatment = treatment == 1
    idx_for_control = treatment == 0

    if estimand == "ate":
        # compute weights for the ATE

        # similar to ifelse(
        # output$treatment == 1, 1 / predictions, 1 / (1 - predictions))
        output.loc[idx_for_treatment] = 1 / propensity_scores.loc[idx_for_treatment]
        output.loc[idx_for_control] = 1 / (1 - propensity_scores.loc[idx_for_control])

    elif estimand == "att":
        # Compute weights for the ATT
        output.loc[idx_for_treatment] = 1  # Weights are 1 for treated individuals

        # Weights for control individuals
        es = propensity_scores.loc[idx_for_control]
        output.loc[idx_for_control] = es / (1 - es)

    else:
        raise ValueError(f"Estimand {estimand} is not supported.")

    return output


def _bootstrapped_ipw_estimator(
    estimand: Literal["ate", "att"],
    propensity_score_estimator: BaseC,
    n_bootstraps: int,
) -> tuple[float, float]:
    """Estimates the average treatment effect (ATE) using the IPW estimator with
    bootstrapping.

    Parameters
    ----------
    estimand : Literal["ate", "att"]
        The estimand of interest. 'ate' for Average Treatment Effect,
        'att' for Average Treatment on the Treated.

    propensity_score_estimator : BaseC
        The propensity score estimator to use.
    """

    pass
