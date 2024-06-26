import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Iterable, Literal
from ...data.datahandler import DataHandler
from ..visualization import plot_obs_vs_pred, decrease_font_sizes_axs
from ...linear.lm import OrdinaryLeastSquares
from ...display.print_utils import print_wrapped
from adjustText import adjust_text


def reverse_argsort(indices):
    n = len(indices)
    reverse_indices = [0] * n
    for i, idx in enumerate(indices):
        reverse_indices[idx] = i
    return reverse_indices


MAX_N_OUTLIERS_TEXT = 20
train_only_message = "This function is only available for training data."


class SingleDatasetLinRegReport:
    """Class for generating regression-relevant diagnostic
    plots and tables for a single linear regression model.
    """

    def __init__(self, model: OrdinaryLeastSquares, dataset: Literal["train", "test"]):
        """
        Initializes a RegressionReport object.

        Parameters
        ----------
        model : BaseRegression.
            The model must already be trained.
        dataset : Literal['train', 'test'].
            The dataset to generate the report for.
        """
        self.model = model

        if dataset == "test":
            self.scorer = model.test_scorer
            self._X_eval_df = self.model._dataemitter.emit_test_Xy()[0]
            self._is_train = False
        elif dataset == "train":
            self.scorer = model.train_scorer
            self._X_eval_df = self.model._dataemitter.emit_train_Xy()[0]
            self._is_train = True
        else:
            raise ValueError('specification must be either "train" or "test".')

        self._y_pred = self.scorer._y_pred
        self._y_true = self.scorer._y_true

        self._residuals = self._y_true - self._y_pred
        self._stdresiduals = self._residuals / np.std(self._residuals)
        self._outlier_threshold = 2
        self._compute_outliers()

        self._include_text = False
        if self._n_outliers <= MAX_N_OUTLIERS_TEXT:
            self._include_text = True

    def plot_obs_vs_pred(
        self,
        show_outliers: bool = True,
        figsize: Iterable = (5, 5),
        ax: plt.Axes = None,
    ) -> plt.Figure:
        """Returns a figure that is a scatter plot of the true and predicted y
        values.

        Parameters
        ----------
        figsize : Iterable.
        ax : plt.Axes.

        Returns
        -------
        - Figure.
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        plot_obs_vs_pred(self._y_pred, self._y_true, figsize, ax)
        if show_outliers and self._n_outliers > 0:
            ax.scatter(
                self._y_pred[self._outliers_residual_mask],
                self._y_true[self._outliers_residual_mask],
                s=2,
                color="red",
            )
            if self._include_text and self._n_outliers <= MAX_N_OUTLIERS_TEXT:
                annotations = []
                for i, label in enumerate(self._outliers_df_idx):
                    annotations.append(
                        ax.annotate(
                            label,
                            (
                                self._y_pred[self._outliers_residual_mask][i],
                                self._y_true[self._outliers_residual_mask][i],
                            ),
                            color="red",
                            fontsize=6,
                        )
                    )
                adjust_text(annotations, ax=ax)

        if fig is not None:
            fig.tight_layout()
            plt.close()
        return fig

    def plot_residuals_vs_fitted(
        self,
        standardized: bool = False,
        show_outliers: bool = True,
        figsize: Iterable = (5, 5),
        ax: plt.Axes = None,
    ) -> plt.Figure:
        """Returns a figure that is a residuals vs fitted (y_pred) plot.

        Parameters
        ----------
        - standardized : bool. If True, standardizes the residuals.
        - show_outliers : bool. If True, plots the outliers in red.
        - figsize : Iterable.
        - ax : Axes

        Returns
        -------
        - Figure
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        residuals = self._residuals
        if standardized:
            residuals = self._stdresiduals

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
        if show_outliers and self._n_outliers > 0:
            ax.scatter(
                self._y_pred[~self._outliers_residual_mask],
                residuals[~self._outliers_residual_mask],
                s=2,
                color="black",
            )
            ax.scatter(
                self._y_pred[self._outliers_residual_mask],
                residuals[self._outliers_residual_mask],
                s=2,
                color="red",
            )
            if self._include_text and self._n_outliers <= MAX_N_OUTLIERS_TEXT:
                annotations = []
                for i, label in enumerate(self._outliers_df_idx):
                    annotations.append(
                        ax.annotate(
                            label,
                            (
                                self._y_pred[self._outliers_residual_mask][i],
                                residuals[self._outliers_residual_mask][i],
                            ),
                            color="red",
                            fontsize=6,
                        )
                    )
                adjust_text(annotations, ax=ax)
        else:
            ax.scatter(self._y_pred, residuals, s=2, color="black")

        ax.set_xlabel("Fitted")
        if standardized:
            ax.set_ylabel("Standardized Residuals")
            ax.set_title("Standardized Residuals vs Fitted")
        else:
            ax.set_ylabel("Residuals")
            ax.set_title("Residuals vs Fitted")
        ax.ticklabel_format(style="sci", axis="both", scilimits=(-3, 3))

        if fig is not None:
            fig.tight_layout()
            plt.close()
        return fig

    def plot_residuals_vs_var(
        self,
        x_var: str,
        standardized: bool = False,
        show_outliers: bool = False,
        figsize: Iterable = (5, 5),
        ax: plt.Axes = None,
    ) -> plt.Figure:
        """Returns a figure that is a residuals vs fitted (y_pred) plot.

        Parameters
        ----------
        x_var : str.
        standardized : bool.
            Default: False. If True, standardizes the residuals.
        show_outliers : bool.
            Default: False. If True, plots the outliers in red.
        figsize : Iterable.
        ax : plt.Axes

        Returns
        -------
        plt.Figure
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        residuals = self._residuals
        if standardized:
            residuals = self._stdresiduals

        x_vals = self._X_eval_df[x_var].to_numpy()

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
        if show_outliers and self._n_outliers > 0:
            ax.scatter(
                x_vals[~self._outliers_residual_mask],
                residuals[~self._outliers_residual_mask],
                s=2,
                color="black",
            )
            ax.scatter(
                x_vals[self._outliers_residual_mask],
                residuals[self._outliers_residual_mask],
                s=2,
                color="red",
            )
            if self._include_text and self._n_outliers <= MAX_N_OUTLIERS_TEXT:
                annotations = []
                for i, label in enumerate(self._outliers_df_idx):
                    annotations.append(
                        ax.annotate(
                            label,
                            (
                                x_vals[self._outliers_residual_mask][i],
                                residuals[self._outliers_residual_mask][i],
                            ),
                            color="red",
                            fontsize=6,
                        )
                    )
                adjust_text(annotations, ax=ax)
        else:
            ax.scatter(x_vals, residuals, s=2, color="black")

        ax.set_xlabel(x_var)
        if standardized:
            ax.set_ylabel("Standardized Residuals")
            ax.set_title(f"Standardized Residuals vs {x_var}")
        else:
            ax.set_ylabel("Residuals")
            ax.set_title(f"Residuals vs {x_var}")
        ax.ticklabel_format(style="sci", axis="both", scilimits=(-3, 3))

        if fig is not None:
            fig.tight_layout()
            plt.close()
        return fig

    def plot_residuals_hist(
        self,
        standardized: bool = False,
        density: bool = False,
        figsize: Iterable = (5, 5),
        ax: plt.Axes = None,
    ) -> plt.Figure:
        """Returns a figure that is a histogram of the residuals.

        Parameters
        ----------
        standardized : bool.
            Default: False. If True, standardizes the residuals.
        density : bool.
            Default: False. If True, plots density rather than frequency.
        figsize : Iterable.
            Default: (5, 5).
        ax : plt.Axes.
            Default: None.

        Returns
        -------
        plt.Figure.
        """
        if density:
            stat = "density"
        else:
            stat = "count"

        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        residuals = self._residuals
        if standardized:
            residuals = self._stdresiduals
        sns.histplot(
            residuals,
            bins="auto",
            color="black",
            edgecolor="none",
            stat=stat,
            ax=ax,
            kde=True,
            alpha=0.2,
        )
        if standardized:
            ax.set_title("Distribution of Standardized Residuals")
            ax.set_xlabel("Standardized Residuals")
        else:
            ax.set_title("Distribution of Residuals")
            ax.set_xlabel("Residuals")
        if density:
            ax.set_ylabel("Density")
        else:
            ax.set_ylabel("Frequency")
        ax.ticklabel_format(style="sci", axis="both", scilimits=(-3, 3))

        if fig is not None:
            fig.tight_layout()
            plt.close()
        return fig

    def plot_scale_location(
        self,
        show_outliers: bool = True,
        figsize: Iterable = (5, 5),
        ax: plt.Axes = None,
    ) -> plt.Figure:
        """Returns a figure that is a plot of the
        sqrt of the residuals versus the fitted.

        Parameters
        ----------
        show_outliers : bool.
            Default: True. If True, plots the outliers in red.
        figsize : Iterable.
            Default: (5, 5).
        ax : plt.Axes.
            Default: None.

        Returns
        -------
        plt.Figure
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        residuals = np.sqrt(np.abs(self._stdresiduals))

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
        if show_outliers and self._n_outliers > 0:
            ax.scatter(
                self._y_pred[~self._outliers_residual_mask],
                residuals[~self._outliers_residual_mask],
                s=2,
                color="black",
            )
            ax.scatter(
                self._y_pred[self._outliers_residual_mask],
                residuals[self._outliers_residual_mask],
                s=2,
                color="red",
            )
            if self._include_text and self._n_outliers <= MAX_N_OUTLIERS_TEXT:
                annotations = []
                for i, label in enumerate(self._outliers_df_idx):
                    annotations.append(
                        ax.annotate(
                            label,
                            (
                                self._y_pred[self._outliers_residual_mask][i],
                                residuals[self._outliers_residual_mask][i],
                            ),
                            color="red",
                            fontsize=6,
                        )
                    )
                adjust_text(annotations, ax=ax)

        else:
            ax.scatter(self._y_pred, residuals, s=2, color="black")

        ax.set_xlabel("Fitted")
        ax.set_ylabel("sqrt(Standardized Residuals)")
        ax.set_title("Scale-Location")
        ax.ticklabel_format(style="sci", axis="both", scilimits=(-3, 3))
        if fig is not None:
            fig.tight_layout()
            plt.close()
        return fig

    def plot_residuals_vs_leverage(
        self,
        standardized: bool = True,
        show_outliers: bool = True,
        figsize: Iterable = (5, 5),
        ax: plt.Axes = None,
    ) -> plt.Figure:
        """Returns a figure that is a plot of the residuals versus leverage.

        Parameters
        ----------
        standardized : bool.
            Default: True. If True, standardizes the residuals.
        show_outliers : bool.
            Default: True. If True, plots the outliers in red.
        figsize : Iterable.
            Default: (5, 5).
        ax : plt.Axes.
            Default: None.

        Returns
        -------
        plt.Figure.
        """
        if not self._is_train:
            print_wrapped(train_only_message, type="WARNING")
            return None

        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        leverage = self.model.estimator._results.get_influence().hat_matrix_diag
        residuals = self._residuals
        if standardized:
            residuals = self._stdresiduals

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
        if show_outliers and self._n_outliers > 0:
            ax.scatter(
                leverage[~self._outliers_residual_mask],
                residuals[~self._outliers_residual_mask],
                s=2,
                color="black",
            )
            ax.scatter(
                leverage[self._outliers_residual_mask],
                residuals[self._outliers_residual_mask],
                s=2,
                color="red",
            )
            if self._include_text and self._n_outliers <= MAX_N_OUTLIERS_TEXT:
                annotations = []
                for i, label in enumerate(self._outliers_df_idx):
                    annotations.append(
                        ax.annotate(
                            label,
                            (
                                leverage[self._outliers_residual_mask][i],
                                residuals[self._outliers_residual_mask][i],
                            ),
                            color="red",
                            fontsize=6,
                        )
                    )
                adjust_text(annotations, ax=ax)

        else:
            ax.scatter(leverage, residuals, s=2, color="black")

        ax.set_xlabel("Leverage")
        if standardized:
            ax.set_ylabel("Standardized Residuals")
            ax.set_title("Standardized Residuals vs Leverage")
        else:
            ax.set_ylabel("Residuals")
            ax.set_title("Residuals vs Leverage")
        ax.ticklabel_format(style="sci", axis="both", scilimits=(-3, 3))

        if fig is not None:
            fig.tight_layout()
            plt.close()
        return fig

    def plot_qq(
        self,
        standardized: bool = True,
        show_outliers: bool = False,
        figsize: Iterable = (5, 5),
        ax: plt.Axes = None,
    ) -> plt.Figure:
        """Returns a quantile-quantile plot.

        Parameters
        ----------
        standardized : bool.
            Default: True. If True, standardizes the residuals.
        show_outliers : bool.
            Default: False. If True, plots the outliers in red.
        figsize : Iterable.
            Default: (5, 5).
        ax : plt.Axes.
            Default: None.

        Returns
        -------
        plt.Figure
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        if standardized:
            residuals = self._stdresiduals
        else:
            residuals = self._residuals

        tup1, tup2 = stats.probplot(residuals, dist="norm")
        theoretical_quantitles, ordered_vals = tup1
        slope, intercept, _ = tup2

        ax.set_title("Q-Q Plot")
        ax.set_xlabel("Theoretical Quantiles")

        if standardized:
            ax.set_ylabel("Standardized Residuals")
        else:
            ax.set_ylabel("Residuals")

        min_val = np.min(theoretical_quantitles)
        max_val = np.max(theoretical_quantitles)
        ax.plot(
            [min_val, max_val],
            [min_val * slope + intercept, max_val * slope + intercept],
            color="gray",
            linestyle="--",
            linewidth=1,
        )

        if show_outliers and self._n_outliers > 0:
            residuals_sorted_idx = reverse_argsort(np.argsort(residuals))

            residuals_df = pd.DataFrame(residuals, columns=["residuals"])
            residuals_df["label"] = self._X_eval_df.index
            residuals_df["is_outlier"] = self._outliers_residual_mask
            residuals_df["theoretical_quantile"] = theoretical_quantitles[
                residuals_sorted_idx
            ]
            residuals_df["ordered_value"] = ordered_vals[residuals_sorted_idx]
            residuals_df_outliers = residuals_df[residuals_df["is_outlier"]]
            residuals_df_not_outliers = residuals_df[~residuals_df["is_outlier"]]

            ax.scatter(
                residuals_df_not_outliers["theoretical_quantile"],
                residuals_df_not_outliers["ordered_value"],
                s=2,
                color="black",
            )
            ax.scatter(
                residuals_df_outliers["theoretical_quantile"],
                residuals_df_outliers["ordered_value"],
                s=2,
                color="red",
            )
            if self._include_text and self._n_outliers <= MAX_N_OUTLIERS_TEXT:
                annotations = []
                for _, row in residuals_df_outliers.iterrows():
                    annotations.append(
                        ax.annotate(
                            row["label"],
                            (row["theoretical_quantile"], row["ordered_value"]),
                            color="red",
                            fontsize=6,
                        )
                    )
                adjust_text(annotations, ax=ax)

        else:
            ax.scatter(theoretical_quantitles, ordered_vals, s=2, color="black")

        if fig is not None:
            fig.tight_layout()
            plt.close()
        return fig

    def plot_diagnostics(
        self, show_outliers: bool = False, figsize: Iterable = (7, 7)
    ) -> plt.Figure:
        """Plots several useful linear regression diagnostic plots.

        Parameters
        ----------
        show_outliers : bool.
            Default: False. If True, plots the residual outliers in red.
        figsize : Iterable.
            Default: (7, 7).

        Returns
        -------
        plt.Figure.
        """
        fig, axs = plt.subplots(2, 2, figsize=figsize)

        self.plot_obs_vs_pred(show_outliers=show_outliers, ax=axs[0][0])
        self.plot_residuals_vs_fitted(show_outliers=show_outliers, ax=axs[0][1])

        if self._is_train:
            self.plot_residuals_vs_leverage(show_outliers=show_outliers, ax=axs[1][0])
        else:
            self.plot_scale_location(show_outliers=show_outliers, ax=axs[1][0])

        self.plot_qq(show_outliers=show_outliers, ax=axs[1][1])

        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        decrease_font_sizes_axs(axs, 5, 5, 0)

        plt.close()
        return fig

    def set_outlier_threshold(self, threshold: float) -> "SingleDatasetLinRegReport":
        """Standardized residuals threshold for outlier identification.
        Recomputes the outliers.

        Parameters
        ----------
        threshold : float.
            Must be a nonnegative value. By default the
            outlier threshold is 2.

        Returns
        -------
        self
        """
        if threshold < 0:
            raise ValueError(
                f"Input threshold must be nonnegative. Received {threshold}."
            )
        self._outlier_threshold = threshold
        self._compute_outliers()
        return self

    def get_outlier_indices(self) -> list:
        """Returns the indices corresponding to DataFrame examples associated
        with standardized residual outliers.

        Returns
        -------
        outliers_df_idx : list ~ (n_outliers).
        """
        return self._outliers_df_idx.tolist()

    def fit_statistics(self) -> pd.DataFrame:
        """Returns a DataFrame containing the goodness-of-fit statistics
        for the model.

        Parameters
        ----------
        pd.DataFrame.
        """
        return self.scorer.stats_df()

    def _compute_outliers(self):
        """Computes the outliers."""
        self._outliers_residual_mask = (
            self._stdresiduals >= self._outlier_threshold
        ) | (self._stdresiduals <= -self._outlier_threshold)
        self._outliers_df_idx = self._X_eval_df.iloc[
            self._outliers_residual_mask
        ].index.to_numpy()
        self._n_outliers = len(self._outliers_df_idx)
        self._include_text = False
        if self._n_outliers <= MAX_N_OUTLIERS_TEXT:
            self._include_text = True


class LinearRegressionReport:
    """LinearRegressionReport.
    Fits the model based on provided DataHandler.
    Wraps train and test SingleDatasetLinRegReport objects.
    """

    def __init__(
        self,
        model: OrdinaryLeastSquares,
        datahandler: DataHandler,
        y_var: str,
        X_vars: Iterable[str],
    ):
        """LinearRegressionReport.
        Fits the model based on provided DataHandler.
        Wraps train and test SingleDatasetLinRegReport objects.

        Parameters
        ----------
        model : OrdinaryLeastSquares.
        datahandler : DataHandler.
            The DataHandler object that contains the data.
        y_var : str.
            The name of the dependent variable.
        X_vars : Iterable[str].
            The names of the independent variables.
        """
        self._model = model
        self._datahandler = datahandler
        self._dataemitter = self._datahandler.train_test_emitter(y_var, X_vars)
        self._model.specify_data(self._dataemitter)
        self._model.fit()

        self._train_report = SingleDatasetLinRegReport(model, "train")
        self._test_report = SingleDatasetLinRegReport(model, "test")

    def train_report(self) -> SingleDatasetLinRegReport:
        """Returns an LinearRegressionReport object for the train dataset

        Returns
        -------
        report : LinearRegressionReport.
        """
        return self._train_report

    def test_report(self) -> SingleDatasetLinRegReport:
        """Returns an LinearRegressionReport object for the test dataset

        Returns
        -------
        report : LinearRegressionReport.
        """
        return self._test_report

    def model(self) -> OrdinaryLeastSquares:
        """Returns the fitted OrdinaryLeastSquares object.

        Returns
        -------
        OrdinaryLeastSquares.
        """
        return self._model

    def fit_statistics(
        self, dataset: Literal["train", "test"] = "test"
    ) -> pd.DataFrame:
        """Returns a DataFrame containing the goodness-of-fit statistics
        for the model.

        Parameters
        ----------
        dataset : Literal['train', 'test'].
            Default: 'test'.

        Returns
        -------
        pd.DataFrame.
        """
        if dataset == "train":
            return self._train_report.fit_statistics()
        else:
            return self._test_report.fit_statistics()

    def stepwise(self) -> "LinearRegressionReport":
        """Performs stepwise selection on the model.

        Parameters
        ----------
        alpha : float.
            Default is 0.05.

        Returns
        -------
        LinearRegressionReport.
        """
        raise NotImplementedError()

    def statsmodels_summary(self):
        """Returns the summary of the statsmodels RegressionResultsWrapper for
        OLS.
        """
        try:
            return self._model.estimator.summary()
        except Exception as e:
            raise RuntimeError(
                "Error occured in statsmodels_summary call. " f"Error: {e}"
            )
