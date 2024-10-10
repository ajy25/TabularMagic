import pandas as pd
from .report import CausalReport
from .causalutils.methods import estimate_ate


class CausalModel:
    """A class for estimating causal effects."""

    def __init__(
        self, data: pd.DataFrame, treatment: str, outcome: str, confounders: list[str]
    ):
        """Initializes a CausalModel object.

        Parameters
        ----------
        data : pd.DataFrame
            The data for which to estimate causal effects.

        treatment : str
            The name of the treatment variable.

        outcome : str
            The name of the outcome variable.

        confounders : list[str]
            The names of the confounding variables.
        """
        self._data = data
        self._treatment = treatment
        self._outcome = outcome
        self._confounders = confounders

    def estimate_ate(self, method: str) -> CausalReport:
        """Estimates the average treatment effect (ATE) using the given method.

        Parameters
        ----------
        method : str
            The method to use for estimating the ATE.
        """
        estimate, se = estimate_ate(
            data=self._data,
            treatment=self._treatment,
            outcome=self._outcome,
            confounders=self._confounders,
            method=method,
        )

        return CausalReport(
            estimate=estimate,
            se=se,
            n_units=self._data.shape[0],
            outcome_var=self._outcome,
            treatment_var=self._treatment,
            confounders=self._confounders,
            estimand="ate",
            method=method,
            method_description="Average treatment effect (ATE) estimated using ",
        )
