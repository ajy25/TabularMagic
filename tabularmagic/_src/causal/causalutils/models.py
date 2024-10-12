import statsmodels.api as sm
import pandas as pd
from ...display.print_utils import suppress_print_output


class _SimpleOLS:
    """Extremely simple OLS wrapper for statsmodels to aid in causal inference."""

    def __init__(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        weights: pd.Series | None = None,
    ):
        """Initializes a _SimpleOLS object.
        Constant term is automatically added to the predictors.

        Parameters
        ----------
        y : pd.Series
            The target variable.

        X : pd.DataFrame
            The predictor variables.
            A constant term is automatically added to the predictors.

        weights : pd.Series | None
            The weights to use for the regression. If None, no weights are used.
            It must be the case that the weights have the same index as the data.
        """
        if not y.index.equals(X.index):
            raise ValueError("The target and predictors must have the same index.")

        with suppress_print_output():
            if weights is not None:
                if not weights.index.equals(X.index):
                    raise ValueError(
                        "The weights index must be the same as the data index."
                    )

                self._sm_results = sm.WLS(
                    endog=y,
                    exog=sm.add_constant(X, has_constant="add"),
                    weights=weights.to_numpy(),
                ).fit(cov_type="HC3")

            else:
                self._sm_results = sm.OLS(
                    endog=y,
                    exog=sm.add_constant(X, has_constant="add"),
                ).fit(cov_type="HC3")

    def get_coef(self, variable: str) -> float:
        """Returns the coefficient for the given variable.

        Parameters
        ----------
        variable : str
            The name of the variable for which to return the coefficient.

        Returns
        -------
        float
            The coefficient for the given variable.
        """

        return self._sm_results.params[variable]

    def get_pvalue(self, variable: str) -> float:
        """Returns the p-value for the given variable.

        Parameters
        ----------
        variable : str
            The name of the variable for which to return the p-value.

        Returns
        -------
        float
            The p-value for the given variable.
        """

        return self._sm_results.pvalues[variable]

    def get_se(self, variable: str) -> float:
        """Returns the standard error for the given variable.

        Parameters
        ----------
        variable : str
            The name of the variable for which to return the standard error.

        Returns
        -------
        float
            The standard error for the given variable.
        """

        return self._sm_results.bse[variable]
