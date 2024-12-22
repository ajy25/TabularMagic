import statsmodels.api as sm
import pandas as pd
from typing import Literal
from ...display.print_utils import suppress_print_output


class SimpleOLS:
    """Extremely simple OLS wrapper for statsmodels to aid in causal inference."""

    def __init__(
        self,
        y: pd.Series,
        X: pd.DataFrame,
        weights: pd.Series | None = None,
        robust: Literal["nonrobust", "HC0", "HC1", "HC2", "HC3"] = "nonrobust",
        weighted_model: Literal["wls", "gaussian_glm"] = "wls",
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

        robust : Literal["HC0", "HC1", "HC2", "HC3"]
            The type of robust standard errors to use.
            If HC0, then the standard errors are not robust.

        weighted_model : Literal["wls", "gaussian_glm"]
            The type of weighted model to use.
            If wls, then a Weighted Least Squares model is used.
            If gaussian_glm, then a Generalized Linear Model with a Gaussian family is used.
        """
        if not y.index.equals(X.index):
            raise ValueError("The target and predictors must have the same index.")

        with suppress_print_output():
            if weights is not None:
                if not weights.index.equals(X.index):
                    raise ValueError(
                        "The weights index must be the same as the data index."
                    )

                if weighted_model == "wls":
                    self._sm_results = sm.WLS(
                        endog=y,
                        exog=sm.add_constant(X, has_constant="add"),
                        weights=weights.to_numpy(),
                    ).fit(cov_type=robust)
                elif weighted_model == "gaussian_glm":
                    self._sm_results = sm.GLM(
                        endog=y,
                        exog=sm.add_constant(X, has_constant="add"),
                        family=sm.families.Gaussian(),
                        freq_weights=weights,
                    ).fit(cov_type=robust)
                else:
                    raise ValueError(
                        "The weighted_model parameter must be either 'wls' "
                        "or 'gaussian_glm'."
                    )

            else:
                self._sm_results = sm.OLS(
                    endog=y,
                    exog=sm.add_constant(X, has_constant="add"),
                ).fit(cov_type=robust)

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
