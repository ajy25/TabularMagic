import statsmodels.api as sm
import pandas as pd
from ...display.print_utils import suppress_print_output


class _CausalLogit:
    """Extremely simple Logit wrapper for statsmodels to aid in causal inference.

    Used for propensity score estimation.
    """

    def __init__(self, data: pd.DataFrame, target: str, predictors: list[str]):
        """Initializes a _CausalLogit object. Performs logistic regression.
        Constant term is automatically added to the predictors.

        Parameters
        ----------
        data : pd.DataFrame
            The data to use for the regression.

        target : str
            The name of the target variable. That is, the Y variable.

        predictors : list[str]
            The names of the predictor variables. That is, the X variables.
            A constant term is automatically added to the predictors.
        """
        self._predictors = predictors
        with suppress_print_output():
            self._sm_results = sm.Logit(
                endog=data[target],
                exog=sm.add_constant(data[predictors], has_constant="add"),
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

    def predict_proba(self, data: pd.DataFrame) -> pd.Series:
        """Predicts the probability of the target variable being 1 for the given data.

        Parameters
        ----------
        data : pd.DataFrame
            The data for which to predict the probability.

        Returns
        -------
        pd.Series
            The predicted probabilities.
        """
        prob_of_one = self._sm_results.predict(
            sm.add_constant(data[self._predictors], has_constant="add")
        )
        return prob_of_one


class _CausalOLS:
    """Extremely simple OLS wrapper for statsmodels to aid in causal inference."""

    def __init__(
        self,
        data: pd.DataFrame,
        target: str,
        predictors: list[str],
        weights: pd.Series | None = None,
    ):
        """Initializes a _CausalOLS object.
        Constant term is automatically added to the predictors.

        Parameters
        ----------
        data : pd.DataFrame
            The data to use for the regression.

        target : str
            The name of the target variable. That is, the Y variable.

        predictors : list[str]
            The names of the predictor variables. That is, the X variables.
            A constant term is automatically added to the predictors.

        weights : pd.Series | None
            The weights to use for the regression. If None, no weights are used.
            It must be the case that the weights have the same index as the data.
        """
        with suppress_print_output():
            if weights is not None:
                if not weights.index.equals(data.index):
                    raise ValueError(
                        "The weights index must be the same as the data index."
                    )

                self._sm_results = sm.WLS(
                    endog=data[target],
                    exog=sm.add_constant(data[predictors], has_constant="add"),
                    weights=weights.to_numpy(),
                ).fit(cov_type="HC3")

            else:
                self._sm_results = sm.OLS(
                    endog=data[target],
                    exog=sm.add_constant(data[predictors], has_constant="add"),
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
