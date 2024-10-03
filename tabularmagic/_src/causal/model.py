import pandas as pd


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
