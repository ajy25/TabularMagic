import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Literal
from ...data.datahandler import DataHandler, DataEmitter
from ...metrics.visualization import plot_obs_vs_pred, decrease_font_sizes_axs
from ...linear.poissonglm import PoissonLinearModel
from ...display.print_utils import print_wrapped
from adjustText import adjust_text
from .linearreport_utils import reverse_argsort, MAX_N_OUTLIERS_TEXT, train_only_message


class SingleDatasetPoisRegReport:
    """Class for generating regression-relevant diagnostic
    plots and tables for a single poisson generalized linear regression model.
    """

    def __init__(self, model: PoissonLinearModel, dataset: Literal["train", "test"]):
        """
        Initializes a SingleDatasetPoisRegReport object.

        Parameters
        ----------
        model : PoissonLinearModel
            The model must already be trained.

        dataset : Literal['train', 'test']
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
        self._pearsonresiduals = (self._y_true - self._y_pred) / np.sqrt(self._y_pred)
        self._outlier_threshold = 2
        self._compute_outliers()
        self._compute_pearson_outliers()

        self._include_text = False
        if self._n_outliers <= MAX_N_OUTLIERS_TEXT:
            self._include_text = True

    def plot_obs_vs_pred(
        self,
        show_outliers: bool = True,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is a scatter plot of the true and predicted y
        values.

        Parameters
        ----------
        show_outliers : bool
            Default: True.
            If True, then the outliers calculated using standard errors will be
            shown in red.

        figsize : tuple[float, float]
            Default: (5.0, 5.0). Sets the size of the resulting graph

        ax : plt.Axes
            Default: None.

        Returns
        -------
        - Figure
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        plot_obs_vs_pred(self._y_pred, self._y_true, self.model._name, figsize, ax)
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
        type: Literal["raw", "standardized", "pearson"] = "raw",
        show_outliers: Literal["none", "standardized", "pearson"] = "none",
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is a residuals vs fitted (y_pred) plot.

        Parameters
        ----------
        type : Literal["raw", "standardized", "pearson"]
            Default: "raw". The type of residuals to be plotted.

        show_outliers : Literal["none", "standardized", "pearson"]
            Default: "none". Colors the outliers red. Outliers are determined by
            the given residual type.

        figsize :  tuple[float, float]
            Default: (5.0, 5.0).

        ax : plt.Axes
            Default: None.

        Returns
        -------
        - Figure
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        if type == "raw":
            residuals = self._residuals
        elif type == "standardized":
            residuals = self._stdresiduals
        elif type == "pearson":
            residuals = self._pearsonresiduals
        else:
            raise ValueError(f"Invalid input for type: {type}")

        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
        if show_outliers == "standardized" and self._n_outliers > 0:
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
        elif show_outliers == "pearson" and self._n_pearson_outliers > 0:
            ax.scatter(
                self._y_pred[~self._outliers_pearson_residual_mask],
                residuals[~self._outliers_pearson_residual_mask],
                s=2,
                color="black",
            )
            ax.scatter(
                self._y_pred[self._outliers_pearson_residual_mask],
                residuals[self._outliers_pearson_residual_mask],
                s=2,
                color="red",
            )
            if (
                self._include_pearson_text
                and self._n_pearson_outliers <= MAX_N_OUTLIERS_TEXT
            ):
                annotations = []
                for i, label in enumerate(self._outliers_pearson_df_idx):
                    annotations.append(
                        ax.annotate(
                            label,
                            (
                                self._y_pred[self._outliers_pearson_residual_mask][i],
                                residuals[self._outliers_pearson_residual_mask][i],
                            ),
                            color="red",
                            fontsize=6,
                        )
                    )
                adjust_text(annotations, ax=ax)
        else:
            ax.scatter(self._y_pred, residuals, s=2, color="black")

        ax.set_xlabel("Fitted")
        if type == "standardized":
            ax.set_ylabel("Standardized Residuals")
            ax.set_title("Standardized Residuals vs Fitted")
        elif type == "pearson":
            ax.set_ylabel("Pearson Residuals")
            ax.set_title("Pearson Residuals vs Fitted")
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
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is a residuals vs fitted (y_pred) plot.

        Parameters
        ----------
        x_var : str
            The predictor variable whose values should be plotted on the x-axis.

        standardized : bool
            Default: False. If True, standardizes the residuals.

        show_outliers : bool
            Default: False. If True, plots the outliers in red.

        figsize : tuple[float, float]
            Default: (5.0, 5.0). Determines the size of the returned figure.

        ax : plt.Axes
            Default: None.

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
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is a histogram of the residuals.

        Parameters
        ----------
        standardized : bool
            Default: False. If True, standardizes the residuals.

        density : bool
            Default: False. If True, plots density rather than frequency.

        figsize : tuple[float, float]
            Default: (5.0, 5.0). Determines the size of the returned figure.

        ax : plt.Axes
            Default: None.

        Returns
        -------
        plt.Figure
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
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is a plot of the
        sqrt of the residuals versus the fitted.

        Parameters
        ----------
        show_outliers : bool
            Default: True. If True, plots the outliers in red.

        figsize : tuple[float, float]
            Default: (5.0, 5.0).

        ax : plt.Axes
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
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is a plot of the residuals versus leverage.

        Parameters
        ----------
        standardized : bool
            Default: True. If True, standardizes the residuals.

        show_outliers : bool
            Default: True. If True, plots the outliers in red.

        figsize : tuple[float, float]
            Default: (5.0, 5.0).

        ax : plt.Axes
            Default: None.

        Returns
        -------
        plt.Figure
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
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a quantile-quantile plot.

        Parameters
        ----------
        standardized : bool
            Default: True. If True, standardizes the residuals.

        show_outliers : bool
            Default: False. If True, plots the outliers in red.

        figsize : tuple[float, float]
            Default: (5.0, 5.0).

        ax : plt.Axes
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
        self, show_outliers: bool = False, figsize: tuple[float, float] = (7, 7)
    ) -> plt.Figure:
        """Plots several useful linear regression diagnostic plots.

        Parameters
        ----------
        show_outliers : bool
            Default: False. If True, plots the residual outliers in red.

        figsize : tuple[float, float]
            Default: (7.0, 7.0).

        Returns
        -------
        plt.Figure
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

    def set_outlier_threshold(self, threshold: float) -> "SingleDatasetPoisRegReport":
        """Standardized residuals threshold for outlier identification.
        Recomputes the outliers.

        Parameters
        ----------
        threshold : float
            Default: 2. Must be a nonnegative value.

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
        self._compute_pearson_outliers()
        return self

    def get_outlier_indices(self) -> list:
        """Returns the indices corresponding to DataFrame examples associated
        with standardized residual outliers.

        Returns
        -------
        outliers_df_idx : list ~ (n_outliers)
        """
        return self._outliers_df_idx.tolist()

    def metrics(self) -> pd.DataFrame:
        """Returns a DataFrame containing the goodness-of-fit statistics
        for the model.

        Returns
        ----------
        pd.DataFrame
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

    def _compute_pearson_outliers(self):
        """Computes the outliers."""
        self._outliers_pearson_residual_mask = (
            self._pearsonresiduals >= self._outlier_threshold
        ) | (self._pearsonresiduals <= -self._outlier_threshold)
        self._outliers_pearson_df_idx = self._X_eval_df.iloc[
            self._outliers_pearson_residual_mask
        ].index.to_numpy()
        self._n_pearson_outliers = len(self._outliers_pearson_df_idx)
        self._include_pearson_text = False
        if self._n_pearson_outliers <= MAX_N_OUTLIERS_TEXT:
            self._include_pearson_text = True


class PoissonRegressionReport:
    """PoissonRegressionReport.
    Fits the model based on provided DataHandler.
    Wraps train and test SingleDatasetPoisRegReport objects.
    """

    def __init__(
        self,
        model: PoissonLinearModel,
        datahandler: DataHandler,
        target: str,
        predictors: list[str],
        dataemitter: DataEmitter | None = None,
    ):
        """PoissonRegressionReport.
        Fits the model based on provided DataHandler.
        Wraps train and test SingleDatasetPoisRegReport objects.

        Parameters
        ----------
        model : PoissonLinearModel

        datahandler : DataHandler
            The DataHandler object that contains the data.

        target : str
            The name of the dependent variable.

        predictors : list[str]
            The names of the independent variables.
        """
        self._model = model
        self._datahandler = datahandler
        if dataemitter is None:
            self._dataemitter = self._datahandler.train_test_emitter(target, predictors)
        else:
            self._dataemitter = dataemitter
        self._model.specify_data(self._dataemitter)
        self._model.fit()
        self._target = target
        self._predictors = predictors
        self._train_report = SingleDatasetPoisRegReport(model, "train")
        self._test_report = SingleDatasetPoisRegReport(model, "test")

    def train_report(self) -> SingleDatasetPoisRegReport:
        """Returns an SingleDatasetPoisRegReport object for the train dataset

        Returns
        -------
        SingleDatasetPoisRegReport
        """
        return self._train_report

    def test_report(self) -> SingleDatasetPoisRegReport:
        """Returns an SingleDatasetPoisRegReport object for the test dataset

        Returns
        -------
        SingleDatasetPoisRegReport
        """
        return self._test_report

    def model(self) -> PoissonLinearModel:
        """Returns the fitted PoissonLinearModel object.

        Returns
        -------
        PoissonLinearModel
        """
        return self._model

    def metrics(self, dataset: Literal["train", "test"] = "test") -> pd.DataFrame:
        """Returns a DataFrame containing the goodness-of-fit statistics
        for the model.

        Parameters
        ----------
        dataset : Literal['train', 'test']
            Default: 'test'.

        Returns
        -------
        pd.DataFrame
        """
        if dataset == "train":
            return self._train_report.metrics()
        else:
            return self._test_report.metrics()

    def step(
        self,
        direction: Literal["both", "backward", "forward"] = "backward",
        criteria: Literal["aic", "bic"] = "aic",
        kept_vars: list[str] | None = None,
        all_vars: list[str] | None = None,
        start_vars: list[str] | None = None,
        max_steps: int = 100,
    ) -> "PoissonRegressionReport":
        """Performs stepwise selection on the model. Returns a new
        PoissonRegressionReport object with the updated model.

        Parameters
        ----------
        direction : Literal["both", "backward", "forward"]
            Default: 'backward'. The direction of the stepwise selection.

        criteria : Literal["aic", "bic"]
            Default: 'aic'. The criteria to use for selecting the best model.

        kept_vars : list[str]
            Default: None. The variables that should be kept in the model.
            If None, defaults to empty list.

        all_vars : list[str]
            Default: None. The variables that are candidates for inclusion in the model.
            If None, defaults to all variables in the training data.

        start_vars : list[str]
            Default: None.
            The variables to start the bidirectional stepwise selection with.
            Ignored if direction is not 'both'. If direction is 'both' and
            start_vars is None, then the starting variables are the kept_vars.

        max_steps : int
            Default: 100. The maximum number of steps to take.

        Returns
        -------
        PoissonRegressionReport
        """
        selected_vars = self._model.step(
            direction=direction,
            criteria=criteria,
            kept_vars=kept_vars,
            all_vars=all_vars,
            start_vars=start_vars,
            max_steps=max_steps,
        )

        new_emitter = self._dataemitter.copy()
        new_emitter.select_predictors(selected_vars)

        return PoissonRegressionReport(
            PoissonLinearModel(),
            self._datahandler,  # only used for y var scaler
            self._target,  # ignored
            self._predictors,  # ignored
            new_emitter,
        )

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

    # Move methods in SingleDatasetPoisRegReport up to LinearRegressionReport
    # to allow useres to call methods from mutliple locations

    def plot_obs_vs_pred(
        self,
        show_outliers: bool = True,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
        dataset: Literal["train", "test"] = "test",
    ) -> plt.Figure:
        """Returns a figure that is a scatter plot of the true and predicted y
        values.

        Parameters
        ----------
        show_outliers : bool
            Default: True.
            If True, then the outliers calculated using standard errors will be
            shown in red.

        figsize : tuple[float, float]
            Default: (5.0,5.0). Sets the size of the resulting graph.

        ax : plt.Axes
            Default: None.

        dataset : Literal['train', 'test']
            Default: 'test'.

        Returns
        -------
        - Figure
        """
        if dataset == "train":
            return self._train_report.plot_obs_vs_pred(
                show_outliers=show_outliers, figsize=figsize, ax=ax
            )
        else:
            return self._test_report.plot_obs_vs_pred(
                show_outliers=show_outliers, figsize=figsize, ax=ax
            )

    def plot_residuals_vs_fitted(
        self,
        standardized: bool = False,
        show_outliers: bool = True,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
        dataset: Literal["train", "test"] = "test",
    ) -> plt.Figure:
        """Returns a figure that is a residuals vs fitted (y_pred) plot.

        Parameters
        ----------
        standardized : bool
            Default: False. If True, plots the standardized residuals as
            opposed to the raw residuals.

        show_outliers : bool
            Default: True. If True, colors the outliers determined by the
            standardized residuals in red.

        figsize : tuple[float, float]
            Default: (5.0, 5.0). Determines the size of the returned figure.

        ax : plt.Axes
            Default = None.

        dataset : Literal['train', 'test']
            Default: 'test'.

        Returns
        -------
        - Figure
        """
        if dataset == "train":
            return self._train_report.plot_residuals_vs_fitted(
                standardized=standardized,
                show_outliers=show_outliers,
                figsize=figsize,
                ax=ax,
            )
        else:
            return self._test_report.plot_residuals_vs_fitted(
                standardized=standardized,
                show_outliers=show_outliers,
                figsize=figsize,
                ax=ax,
            )

    def plot_residuals_vs_var(
        self,
        x_var: str,
        standardized: bool = False,
        show_outliers: bool = False,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
        dataset: Literal["train", "test"] = "test",
    ) -> plt.Figure:
        """Returns a figure that is a residuals vs fitted (y_pred) plot.

        Parameters
        ----------
        x_var : str
            The predictor variable whose values should be plotted on the x-axis.

        standardized : bool
            Default: False. If True, standardizes the residuals.

        show_outliers : bool
            Default: False. If True, plots the outliers in red.

        figsize : tuple[float, float]
            Default: (5.0, 5.0). Determines the size of the returned figure.

        ax : plt.Axes
            Default: None.

        dataset : Literal['train', 'test']
            Default: 'test'.

        Returns
        -------
        plt.Figure
        """
        if dataset == "train":
            return self._train_report.plot_residuals_vs_var(
                x_var=x_var,
                standardized=standardized,
                show_outliers=show_outliers,
                figsize=figsize,
                ax=ax,
            )
        else:
            return self._test_report.plot_residuals_vs_var(
                x_var=x_var,
                standardized=standardized,
                show_outliers=show_outliers,
                figsize=figsize,
                ax=ax,
            )

    def plot_residuals_hist(
        self,
        standardized: bool = False,
        density: bool = False,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
        dataset: Literal["train", "test"] = "test",
    ) -> plt.Figure:
        """Returns a figure that is a histogram of the residuals.

        Parameters
        ----------
        standardized : bool
            Default: False. If True, standardizes the residuals.

        density : bool
            Default: False. If True, plots density rather than frequency.

        figsize : tuple[float, float]
            Default: (5.0, 5.0). Determines the size of the returned figure.

        ax : plt.Axes
            Default: None.

        dataset : Literal['train', 'test']
            Default: 'test'.

        Returns
        -------
        plt.Figure
        """
        if dataset == "train":
            return self._train_report.plot_residuals_hist(
                standardized=standardized, density=density, figsize=figsize, ax=ax
            )
        else:
            return self._test_report.plot_residuals_hist(
                standardized=standardized, density=density, figsize=figsize, ax=ax
            )

    def plot_scale_location(
        self,
        show_outliers: bool = True,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
        dataset: Literal["train", "test"] = "test",
    ) -> plt.Figure:
        """Returns a figure that is a plot of the
        sqrt of the residuals versus the fitted.

        Parameters
        ----------
        show_outliers : bool
            Default: True. If True, plots the outliers in red.

        figsize : tuple[float, float]
            Default: (5.0, 5.0).

        ax : plt.Axes
            Default: None.

        dataset : Literal['train', 'test']
            Default: 'test'.

        Returns
        -------
        plt.Figure
        """
        if dataset == "train":
            return self._train_report.plot_scale_location(
                show_outliers=show_outliers, figsize=figsize, ax=ax
            )
        else:
            return self._test_report.plot_scale_location(
                show_outliers=show_outliers, figsize=figsize, ax=ax
            )

    def plot_residuals_vs_leverage(
        self,
        standardized: bool = True,
        show_outliers: bool = True,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
        dataset: Literal["train", "test"] = "test",
    ) -> plt.Figure:
        """Returns a figure that is a plot of the residuals versus leverage.

        Parameters
        ----------
        standardized : bool
            Default: True. If True, standardizes the residuals.

        show_outliers : bool
            Default: True. If True, plots the outliers in red.

        figsize : tuple[float, float]
            Default: (5.0, 5.0).

        ax : plt.Axes
            Default: None.

        dataset : Literal['train', 'test']
            Default: 'test'.

        Returns
        -------
        plt.Figure
        """
        if dataset == "train":
            return self._train_report.plot_residuals_vs_leverage(
                standardized=standardized,
                show_outliers=show_outliers,
                figsize=figsize,
                ax=ax,
            )
        else:
            return self._test_report.plot_residuals_vs_leverage(
                standardized=standardized,
                show_outliers=show_outliers,
                figsize=figsize,
                ax=ax,
            )

    def plot_qq(
        self,
        standardized: bool = True,
        show_outliers: bool = False,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
        dataset: Literal["train", "test"] = "test",
    ) -> plt.Figure:
        """Returns a quantile-quantile plot.

        Parameters
        ----------
        standardized : bool
            Default: True. If True, standardizes the residuals.

        show_outliers : bool
            Default: False. If True, plots the outliers in red.

        figsize : tuple[float, float]
            Default: (5.0, 5.0).

        ax : plt.Axes
            Default: None.

        dataset : Literal['train', 'test']
            Default: 'test'.

        Returns
        -------
        plt.Figure
        """
        if dataset == "train":
            return self._train_report.plot_qq(
                standardized=standardized,
                show_outliers=show_outliers,
                figsize=figsize,
                ax=ax,
            )
        else:
            return self._test_report.plot_qq(
                standardized=standardized,
                show_outliers=show_outliers,
                figsize=figsize,
                ax=ax,
            )

    def plot_diagnostics(
        self,
        show_outliers: bool = False,
        figsize: tuple[float, float] = (7.0, 7.0),
        dataset: Literal["train", "test"] = "test",
    ) -> plt.Figure:
        """Plots several useful linear regression diagnostic plots.

        Parameters
        ----------
        show_outliers : bool
            Default: False. If True, plots the residual outliers in red.

        figsize : tuple[float, float]
            Default: (7.0, 7.0).

        dataset : Literal['train', 'test']
            Default: 'test'.

        Returns
        -------
        plt.Figure
        """
        if dataset == "train":
            return self._train_report.plot_diagnostics(
                show_outliers=show_outliers, figsize=figsize
            )
        else:
            return self._test_report.plot_diagnostics(
                show_outliers=show_outliers, figsize=figsize
            )

    def set_outlier_threshold(
        self, threshold: float, dataset: Literal["train", "test"] = "test"
    ) -> "SingleDatasetPoisRegReport":
        """Standardized residuals threshold for outlier identification.
        Recomputes the outliers.

        Parameters
        ----------
        threshold : float
            Default: 2. Must be a nonnegative value.

        dataset : Literal['train', 'test']
            Default: 'test'.

        Returns
        -------
        self
        """
        if dataset == "train":
            self._train_report.set_outlier_threshold(threshold=threshold)
        else:
            self._test_report.set_outlier_threshold(threshold=threshold)

    def get_outlier_indices(self, dataset: Literal["train", "test"] = "test") -> list:
        """Returns the indices corresponding to DataFrame examples associated
        with standardized residual outliers.

        Parameters
        ----------
        dataset : Literal['train', 'test']
            Default: 'test'.

        Returns
        -------
        outliers_df_idx : list ~ (n_outliers)
        """
        if dataset == "train":
            return self._train_report.get_outlier_indices()
        else:
            return self._test_report.get_outlier_indices()

    def _compute_outliers(self, dataset: Literal["train", "test"] = "test"):
        """Computes the outliers.

        Parameters
        ----------
        dataset : Literal['train', 'test']
            Default: 'test'.
        """
        if dataset == "train":
            return self._train_report._compute_outliers()
        else:
            return self._test_report._compute_outliers()
