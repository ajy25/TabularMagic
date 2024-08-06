import pandas as pd
from typing import Literal
import matplotlib.pyplot as plt
from ...display.print_utils import suppress_stdout
from ...data.datahandler import DataHandler, DataEmitter
from .negbinglmreg import SingleDatasetNegBinRegReport
from .poissonglmreg import SingleDatasetPoisRegReport
from ..countglm import CountLinearModel


class CountRegressionReport:
    """CountRegressionReport.
    Fits the model based on provided DataHandler.
    Wraps train and test SingleDatasetPoisRegReport or SingleDatasetNegBinReport
    objects depending on which model is chosen.
    """

    def __init__(
        self,
        model: CountLinearModel,
        datahandler: DataHandler,
        target: str,
        predictors: list[str],
        dataemitter: DataEmitter | None = None,
    ):
        """CountRegressionReport.
        Fits the model based on provided DataHandler.
        Wraps train and test SingleDatasetPoisRegReport or
        SingleDatasetNegBinRegReport objects depending on chosen model.

        Parameters
        ----------
        model : CountLinearModel

        datahandler : DataHandler
            The DataHandler object that contains the data.

        target : str
            The name of the dependent variable.

        predictors : list[str]
            The names of the independent variables.

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
        if self._model._type == "poisson":
            self._train_report = SingleDatasetPoisRegReport(model, "train")
            self._test_report = SingleDatasetPoisRegReport(model, "test")
        else:
            self._train_report = SingleDatasetNegBinRegReport(model, "train")
            self._test_report = SingleDatasetNegBinRegReport(model, "test")

    def train_report(self) -> SingleDatasetPoisRegReport | SingleDatasetNegBinRegReport:
        """Returns a SingleDatasetPoisRegReport or a SingleDatasetNegBinRegReport
        object for the train dataset depending on the statistical test
        for overdispersion.

        Returns
        -------
        SingleDatasetPoisRegReport | SingleDatasetNegBinRegReport
        """
        return self._train_report

    def test_report(self) -> SingleDatasetPoisRegReport | SingleDatasetNegBinRegReport:
        """Returns a SingleDatasetPoisRegReport or a SingleDatasetNegBinRegReport
        object for the test dataset depending on the statistical test
        for overdispersion.

        Returns
        -------
        SingleDatasetPoisRegReport | SingleDatasetNegBinRegReport
        """
        return self._test_report

    def model(self) -> CountLinearModel:
        """Returns the fitted CountLinearModel
        object.

        Returns
        -------
        CountLinearModel
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
    ) -> "CountRegressionReport":
        """Performs stepwise selection on the model. Returns a new
        CountRegressionReport object with the updated model.

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
        CountRegressionReport
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

        return CountRegressionReport(
            CountLinearModel(),
            self._datahandler,
            self._target,
            self._predictors,
            dataemitter=new_emitter,
        )

    def statsmodels_summary(self):
        """Returns the summary of the statsmodels RegressionResultsWrapper for
        negative binomial or poisson glm.
        """
        try:
            return self._model.estimator.summary()
        except Exception as e:
            raise RuntimeError(
                "Error occured in statsmodels_summary call. " f"Error: {e}"
            )

    # Move methods in SingleDatasetPoisRegReport/SingleDatasetNegBinRegReport
    # up to LinearRegressionReport to allow useres to call methods from
    # mutliple locations

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
    ) -> SingleDatasetPoisRegReport | SingleDatasetNegBinRegReport:
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
