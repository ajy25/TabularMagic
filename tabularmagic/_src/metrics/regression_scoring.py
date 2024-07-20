import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from ..._src.display.print_utils import print_wrapped


class RegressionScorer:
    """Class for scoring of regression fits.
    Only inputs are predicted and true values.
    Capable of scoring cross validation outputs.

    RegressionScorer is indexable by integers in the following order:
        (MSE, MAD, Pearson Correlation, Spearman Correlation, R Squared,
        Adjusted R Squared).

    RegressionScorer also is indexable by a string key similar to the
    dictionary: {'mse': MSE, 'mad': MAD, 'pearsonr': Pearson Correlation,
        'spearmanr': Spearman Correlation, 'r2': R Squared,
        'adjr2': Adjusted R Squared}.
    """

    def __init__(
        self,
        y_pred: np.ndarray | list,
        y_true: np.ndarray | list,
        n_predictors: int | None = None,
        name: str | None = None,
    ):
        """
        Initializes a RegressionScorer object.

        Parameters
        ----------
        y_pred : np.ndarray ~ (sample_size) | list[np.ndarray ~ (sample_size)].
        y_true : np.ndarray ~ (sample_size) | list[np.ndarray ~ (sample_size)].
        n_predictors : int.
            Default: None.
        name : str.
            Default: None.
        """
        if name is None:
            self._name = "Model"
        else:
            self._name = name
        self._n_predictors = n_predictors
        self._y_pred = y_pred
        self._y_true = y_true

        self._stats_df = None
        self._cv_stats_df = None
        self._set_stats_df()

    def _set_stats_df(self):
        """Sets the stats_df attribute of the RegressionScorer object."""

        y_pred = self._y_pred
        y_true = self._y_true

        if isinstance(y_pred, np.ndarray) and isinstance(y_true, np.ndarray):
            n = len(y_pred)
            df = pd.DataFrame(columns=[self._name])
            df.loc["rmse", self._name] = root_mean_squared_error(y_true, y_pred)
            df.loc["mad", self._name] = mean_absolute_error(y_true, y_pred)
            df.loc["pearsonr", self._name] = pearsonr(y_true, y_pred)[0]
            df.loc["spearmanr", self._name] = spearmanr(y_true, y_pred)[0]
            df.loc["r2", self._name] = r2_score(y_true, y_pred)
            if self._n_predictors is None:
                df.loc["adjr2", self._name] = np.NaN
            else:
                try:
                    adjr2 = 1 - (
                        ((1 - df.loc["r2", self._name]) * (n - 1))
                        / (n - self._n_predictors - 1)
                    )
                except ZeroDivisionError:
                    adjr2 = np.NaN
                    print_wrapped(
                        "ZeroDivisionError occurred during calculation of "
                        "adjusted R squared. Setting adjusted R squared to NaN.",
                        type="WARNING",
                    )
            df.loc["n_obs", self._name] = len(y_true)
            df = df.rename_axis("Statistic", axis="rows")
            self._stats_df = df
        elif isinstance(y_pred, list) and isinstance(y_true, list):
            assert len(y_pred) == len(y_true)
            cvdf = pd.DataFrame(columns=["Fold", "Statistic", self._name])
            for i, (y_pred_elem, y_true_elem) in enumerate(zip(y_pred, y_true)):
                n = len(y_pred_elem)
                cvdf.loc[len(cvdf)] = pd.Series(
                    {
                        "Statistic": "rmse",
                        self._name: root_mean_squared_error(y_true_elem, y_pred_elem),
                        "Fold": i,
                    }
                )
                cvdf.loc[len(cvdf)] = pd.Series(
                    {
                        "Statistic": "mad",
                        self._name: mean_absolute_error(y_true_elem, y_pred_elem),
                        "Fold": i,
                    }
                )
                cvdf.loc[len(cvdf)] = pd.Series(
                    {
                        "Statistic": "pearsonr",
                        self._name: pearsonr(y_true_elem, y_pred_elem)[0],
                        "Fold": i,
                    }
                )
                cvdf.loc[len(cvdf)] = pd.Series(
                    {
                        "Statistic": "spearmanr",
                        self._name: spearmanr(y_true_elem, y_pred_elem)[0],
                        "Fold": i,
                    }
                )
                r2 = r2_score(y_true_elem, y_pred_elem)
                cvdf.loc[len(cvdf)] = pd.Series(
                    {"Statistic": "r2", self._name: r2, "Fold": i}
                )
                if self._n_predictors is None:
                    adjr2 = np.NaN
                else:
                    try:
                        adjr2 = 1 - (
                            ((1 - r2) * (n - 1)) / (n - self._n_predictors - 1)
                        )
                    except ZeroDivisionError:
                        adjr2 = np.NaN
                        print_wrapped(
                            "ZeroDivisionError occurred during calculation of "
                            "adjusted R squared. Setting adjusted R squared to NaN.",
                            type="WARNING",
                        )

                cvdf.loc[len(cvdf)] = pd.Series(
                    {"Statistic": "adjr2", self._name: adjr2, "Fold": i}
                )
                cvdf.loc[len(cvdf)] = pd.Series(
                    {"Statistic": "n_obs", self._name: n, "Fold": i}
                )
            self._cv_stats_df = cvdf.set_index(["Fold", "Statistic"])
            self._stats_df = (
                cvdf.groupby(["Statistic"])[[self._name]]
                .mean()
                .reindex(
                    ["rmse", "mad", "pearsonr", "spearmanr", "r2", "adjr2", "n_obs"]
                )
            )

        else:
            raise ValueError("Input types for y_pred and y_true are invalid.")

    def stats_df(self) -> pd.DataFrame:
        """Outputs a DataFrame that contains the fit statistics.

        Returns
        -------
        pd.DataFrame.
        """
        return self._stats_df

    def cv_stats_df(self) -> pd.DataFrame:
        """Outputs a DataFrame that contains the cross validation statistics.

        Returns
        -------
        pd.DataFrame.
        """
        return self._cv_stats_df
