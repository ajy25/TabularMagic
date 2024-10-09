import numpy as np
import pandas as pd
from typing import Literal

from .models import _CausalLogit, _CausalOLS


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
    idx_for_treatment = (treatment == 1).index
    idx_for_control = (treatment == 0).index

    if estimand == "ate":
        # compute weights for the ATE
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


def estimate_ate(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: list[str],
    method: Literal[
        "naive", "outcome_regression", "weighted_regression", "ipw_estimator"
    ],
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

            The estimate is the difference in means between the treatment and
            control groups. The standard error of the estimate is computed
            using the formula for the standard error of the difference in means.

        - "outcome_regression":
            Estimates the ATE using linear regression (OLS). Assumes no
            interaction between the treatment variable and the confounders.

            The estimate and the SE of the estimate are given by `statsmodels.OLS`.

        - "weighted_regression":
            Estimates the ATE using weighted linear regression. Weights the
            observations by the inverse of the propensity score. Propensity
            scores are estimated using a logistic regression model (predict
            probability of treatment given confounders).
            Weighted regression predicts the outcome using the treatment and
            confounders, and weights the observations by the inverse of the
            propensity score.

            The estimate and the SE of the estimate are given by `statsmodels.WLS`.

        - "ipw_estimator":
            Estimates the ATE using the inverse propensity weighting (IPW)
            estimator. Propensity scores are estimated using a logistic
            regression model (predict probability of treatment given
            confounders). The ATE is estimated as the weighted difference in
            outcomes between the treatment and control groups, where the
            weights are the inverse of the propensity scores.

            The estimate is the weighted difference in outcomes between the
            treatment and control groups. The standard error of the estimate is
            computed using the formula for the standard error of the weighted
            difference in means.


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

    elif method == "outcome_regression":
        # Fits an OLS model with the treatment variable and confounders.
        # Interprets the coefficient of the treatment variable as the ATE.
        model = _CausalOLS(
            data=data, target=outcome, predictors=[treatment] + confounders
        )
        ate = model.get_coef(treatment)
        se = model.get_se(treatment)
        return ate, se

    elif method == "weighted_regression":
        # Fits a logistic regression model to estimate the propensity score.
        # Weights the observations by the inverse of the propensity score.
        # Fits a weighted OLS model with the treatment variable and confounders.
        # Interprets the coefficient of the treatment variable as the ATE.
        propensity_model = _CausalLogit(
            data=data, target=treatment, predictors=confounders
        )
        propensity_scores = propensity_model.predict_proba(data)
        weights_series = compute_weights_from_propensity_scores(
            estimand="ate",
            propensity_scores=propensity_scores,
            treatment=data[treatment],
        )
        model = _CausalOLS(
            data=data,
            target=outcome,
            predictors=[treatment] + confounders,
            weights=weights_series,
        )
        ate = model.get_coef(treatment)
        se = model.get_se(treatment)
        return ate, se

    elif method == "ipw_estimator":
        # Fits a logistic regression model to estimate the propensity score.
        # Weights the observations by the inverse of the propensity score.
        # Estimates the ATE as the weighted difference in outcomes between the
        # treatment and control groups.
        propensity_model = _CausalLogit(
            data=data, target=treatment, predictors=confounders
        )
        propensity_scores = propensity_model.predict_proba(data)
        weights_series = compute_weights_from_propensity_scores(
            estimand="ate",
            propensity_scores=propensity_scores,
            treatment=data[treatment],
        )

        outcome = data[outcome].to_numpy()
        treatment = data[treatment].to_numpy()
        weights = weights_series.to_numpy()

        ate = np.mean(
            outcome * treatment * weights - outcome * (1 - treatment) * weights
        )
        # compute the sandwich estimator for the standard error
        se = np.sqrt(
            np.var(outcome * treatment * weights - outcome * (1 - treatment) * weights)
            / len(outcome)
        )

    else:
        raise ValueError(f"Method {method} is not supported.")
