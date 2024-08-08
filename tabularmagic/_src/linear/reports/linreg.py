import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Literal
import warnings
from adjustText import adjust_text
from ...data import DataHandler, DataEmitter
from ...metrics.visualization import plot_obs_vs_pred, decrease_font_sizes_axs
from ..lm import OLSLinearModel
from ...display.print_utils import print_wrapped
from .linearreport_utils import reverse_argsort, MAX_N_OUTLIERS_TEXT, train_only_message
from ...exploratory.stattests import StatisticalTestResult


class SingleDatasetLinRegReport:
    """Class for generating regression-relevant diagnostic
    plots and tables for a single linear regression model.
    """

    def __init__(self, model: OLSLinearModel, dataset: Literal["train", "test"]):
        """
        Initializes a SingleDatasetLinRegReport object.

        Parameters
        ----------
        model : OLSLinearModel.
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
        self._outlier_threshold = 2
        self._compute_outliers()

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
            Default: (5.0,5.0). Sets the size of the resulting graph.

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
        standardized: bool = False,
        show_outliers: bool = True,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
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
        predictor: str,
        standardized: bool = False,
        show_outliers: bool = False,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is a residuals vs fitted (y_pred) plot.

        Parameters
        ----------
        predictor : str
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

        x_vals = self._X_eval_df[predictor].to_numpy()

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

        ax.set_xlabel(predictor)
        if standardized:
            ax.set_ylabel("Standardized Residuals")
            ax.set_title(f"Standardized Residuals vs {predictor}")
        else:
            ax.set_ylabel("Residuals")
            ax.set_title(f"Residuals vs {predictor}")
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
        self, show_outliers: bool = False, figsize: tuple[float, float] = (7.0, 7.0)
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

    def set_outlier_threshold(self, threshold: float) -> "SingleDatasetLinRegReport":
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


class LinearRegressionReport:
    """LinearRegressionReport.
    Fits the model based on provided DataHandler.
    Wraps train and test SingleDatasetLinRegReport objects.
    """

    def __init__(
        self,
        model: OLSLinearModel,
        datahandler: DataHandler,
        target: str,
        predictors: list[str],
        dataemitter: DataEmitter | None = None,
    ):
        """LinearRegressionReport.
        Fits the model based on provided DataHandler.
        Wraps train and test SingleDatasetLinRegReport objects.

        Parameters
        ----------
        model : OrdinaryLeastSquares

        datahandler : DataHandler
            The DataHandler object that contains the data.

        target : str
            The name of the target variable.

        predictors : list[str]
            The names of the predictor variables.

        dataemitter : DataEmitter
            Default: None. The DataEmitter object that emits the data.
            Optionally you can initialize the report with a DataEmitter object
            instead of a DataHandler object. If not None, will ignore the
            values of target and predictors.
        """
        self._model = model
        self._datahandler = datahandler

        if dataemitter is not None:
            self._dataemitter = dataemitter
        else:
            self._dataemitter = self._datahandler.train_test_emitter(target, predictors)
        self._model.specify_data(self._dataemitter)
        self._model.fit()
        self._target = target
        self._predictors = predictors
        self._train_report = SingleDatasetLinRegReport(model, "train")
        self._test_report = SingleDatasetLinRegReport(model, "test")

    def _train_report(self) -> SingleDatasetLinRegReport:
        """Returns an SingleDatasetLinRegReport object for the train dataset

        Returns
        -------
        SingleDatasetLinRegReport
        """
        return self._train_report

    def _test_report(self) -> SingleDatasetLinRegReport:
        """Returns an SingleDatasetLinRegReport object for the test dataset

        Returns
        -------
        SingleDatasetLinRegReport
        """
        return self._test_report

    def model(self) -> OLSLinearModel:
        """Returns the fitted OLSLinearModel object.

        Returns
        -------
        OLSLinearModel
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
    ) -> "LinearRegressionReport":
        """Performs stepwise selection on the model. Returns a new
        LinearRegressionReport object with the updated model.

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
        LinearRegressionReport
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

        return LinearRegressionReport(
            OLSLinearModel(),
            self._datahandler,  # only used for y var scaler
            self._target,  # ignored
            self._predictors,  # ignored
            new_emitter,
        )

    def test_lr(
        self, alternative_report: "LinearRegressionReport"
    ) -> StatisticalTestResult:
        """Performs a likelihood ratio test to compare an alternative
        OLSLinearModel. Returns an object of class StatisticalTestResult
        describing the results.

        Parameters
        ----------
        alternative_report : LinearRegressionReport
            The report of an alternative OLSLinearModel. The alternative
            model must be a nested version of the current model or vice-versa.

        Returns
        -------
        StatisticalTestResult
        """
        # Determine which report is the reduced model

        # Get the models from each report
        original_model = self._train_report.model.estimator
        alternative_model = alternative_report._train_report().model.estimator

        # Get the number of predictors for each model
        num_predictors_orig = len(self._train_report._X_eval_df.columns)
        num_predictors_alternative = len(
            alternative_report._train_report()._X_eval_df.columns
        )

        if num_predictors_orig > num_predictors_alternative:
            full_model = original_model
            reduced_model = alternative_model
        elif num_predictors_orig < num_predictors_alternative:
            full_model = alternative_model
            reduced_model = original_model
        else:
            # Raise an error if the number of predictors are the same
            raise ValueError("One model must be a reduced version of the other")

        # Raise ValueError if one set of predictors is not a subset of the other
        orig_var_set = set(self._train_report._X_eval_df.columns)
        alt_var_set = set(alternative_report._train_report()._X_eval_df.columns)

        if not (orig_var_set < alt_var_set or orig_var_set > alt_var_set):
            raise ValueError("One model must be a reduced version of the other")

        # Extract the results of the test and temporarily suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            lr_stat, p_value, dr_diff = full_model.compare_lr_test(reduced_model)

        # Initialize and return an object of class StatisticalTestResult
        lr_result = StatisticalTestResult(
            description="Likelihood Ratio Test",
            statistic=lr_stat,
            pval=p_value,
            degfree=dr_diff,
            statistic_description="Chi-square",
            null_hypothesis_description="The full model does not fit the "
            "data significantly better than the reduced model",
            alternative_hypothesis_description="The full model fits the "
            "data signficantly better than the reduced model",
            assumptions_description="The data must be homoscedastic and "
            "uncorrelated",
        )

        return lr_result

    def test_partialf(self, alternative_report):
        """Performs a partial F-test to compare an alternative OLSLinearModel.
        Returns an object of class StatisticalTestResult describing the results.

        Parameters
        ----------
        alternative_report : LinearRegressionReport
            The report of an alternative OLSLinearModel. The alternative
            model must be a nested version of the current model or vice-versa.

        Returns
        -------
        StatisticalTestResult
        """
        # Determine which report is the reduced model

        # Get the models from each report
        original_model = self._train_report.model.estimator
        alternative_model = alternative_report.train_report().model.estimator

        # Get the number of predictors for each model
        num_predictors_orig = len(self._train_report._X_eval_df.columns)
        num_predictors_alternative = len(
            alternative_report.train_report()._X_eval_df.columns
        )

        if num_predictors_orig > num_predictors_alternative:
            full_model = original_model
            reduced_model = alternative_model
        elif num_predictors_orig < num_predictors_alternative:
            full_model = alternative_model
            reduced_model = original_model
        else:
            # Raise an error if the number of predictors are the same
            raise ValueError("One model must be a reduced version of the other")

        # Raise ValueError if one set of predictors is not a subset of the other
        orig_var_set = set(self._train_report._X_eval_df.columns)
        alt_var_set = set(alternative_report.train_report()._X_eval_df.columns)

        if not (orig_var_set < alt_var_set or orig_var_set > alt_var_set):
            raise ValueError("One model must be a reduced version of the other")

        # Extract the results of the test and suppress warnings temporarily
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            f_value, p_value, dr_diff = full_model.compare_f_test(reduced_model)

        # Initialize and return an object of class StatisticalTestResult
        partial_f_result = StatisticalTestResult(
            description="Partial F-Test",
            statistic=f_value,
            pval=p_value,
            degfree=dr_diff,
            statistic_description="F-statistic",
            null_hypothesis_description="The coefficients of the additional "
            "predictors are all zero",
            alternative_hypothesis_description="At least one of the "
            "coefficients of the additional predictors is not zero",
            assumptions_description="The data must be homoscedastic and "
            "have no autocorrelation",
        )

        return partial_f_result

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

    # Move methods in SingleDatasetLinRegReport up to LinearRegressionReport
    # to allow useres to call methods from mutliple locations

    def plot_obs_vs_pred(
        self,
        dataset: Literal["train", "test"] = "test",
        show_outliers: bool = True,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is a scatter plot of the true and predicted y
        values.

        Parameters
        ----------
        dataset : Literal['train', 'test']
            Default: 'test'.

        show_outliers : bool
            Default: True.
            If True, then the outliers calculated using standard errors will be
            shown in red.

        figsize : tuple[float, float]
            Default: (5.0,5.0). Sets the size of the resulting graph.

        ax : plt.Axes
            Default: None.

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
        dataset: Literal["train", "test"] = "test",
        standardized: bool = False,
        show_outliers: bool = True,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is a residuals vs fitted (y_pred) plot.

        Parameters
        ----------
        dataset : Literal['train', 'test']
            Default: 'test'.

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
        predictor: str,
        dataset: Literal["train", "test"] = "test",
        standardized: bool = False,
        show_outliers: bool = False,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is a residuals vs fitted (y_pred) plot.

        Parameters
        ----------
        predictor : str
            The predictor variable whose values should be plotted on the x-axis.

        dataset : Literal['train', 'test']
            Default: 'test'.

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
        if dataset == "train":
            return self._train_report.plot_residuals_vs_var(
                predictor=predictor,
                standardized=standardized,
                show_outliers=show_outliers,
                figsize=figsize,
                ax=ax,
            )
        else:
            return self._test_report.plot_residuals_vs_var(
                predictor=predictor,
                standardized=standardized,
                show_outliers=show_outliers,
                figsize=figsize,
                ax=ax,
            )

    def plot_residuals_hist(
        self,
        dataset: Literal["train", "test"] = "test",
        standardized: bool = False,
        density: bool = False,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is a histogram of the residuals.

        Parameters
        ----------
        dataset : Literal['train', 'test']
            Default: 'test'.

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
        dataset: Literal["train", "test"] = "test",
        show_outliers: bool = True,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is a plot of the
        sqrt of the residuals versus the fitted.

        Parameters
        ----------
        dataset : Literal['train', 'test']
            Default: 'test'.

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
        dataset: Literal["train", "test"] = "test",
        standardized: bool = True,
        show_outliers: bool = True,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is a plot of the residuals versus leverage.

        Parameters
        ----------
        dataset : Literal['train', 'test']
            Default: 'test'.

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
        dataset: Literal["train", "test"] = "test",
        standardized: bool = True,
        show_outliers: bool = False,
        figsize: tuple[float, float] = (5.0, 5.0),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a quantile-quantile plot.

        Parameters
        ----------
        dataset : Literal['train', 'test']
            Default: 'test'.

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
        dataset: Literal["train", "test"] = "test",
        show_outliers: bool = False,
        figsize: tuple[float, float] = (7.0, 7.0),
    ) -> plt.Figure:
        """Plots several useful linear regression diagnostic plots.

        Parameters
        ----------
        dataset : Literal['train', 'test']
            Default: 'test'.

        show_outliers : bool
            Default: False. If True, plots the residual outliers in red.

        figsize : tuple[float, float]
            Default: (7.0, 7.0).

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
        self, 
        threshold: float, 
        dataset: Literal["train", "test"] = "test"
    ) -> "SingleDatasetLinRegReport":
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
