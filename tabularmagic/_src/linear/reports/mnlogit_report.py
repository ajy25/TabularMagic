import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Literal
from adjustText import adjust_text
from ...data.datahandler import DataHandler, DataEmitter
from ...metrics.visualization import plot_obs_vs_pred, decrease_font_sizes_axs
from ..mnlogit import MNLogitLinearModel
from ...display.print_utils import print_wrapped, suppress_print_output
from .linearreport_utils import reverse_argsort, MAX_N_OUTLIERS_TEXT, train_only_message


class _SingleDatasetMNLogitReport:
    """Class for generating regression-relevant diagnostic
    plots and tables for a MNLogit model.
    """

    def __init__(self, model: MNLogitLinearModel, dataset: Literal["train", "test"]):
        """
        Initializes a _SingleDatasetMNLogitReport object.

        Parameters
        ----------
        model : MNLogitLinearModel
            The model must already be trained.

        dataset : Literal['train', 'test']
            The dataset to generate the report for.
        """
        self.model = model

        with suppress_print_output():
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

        self._y_pred_score = self.scorer._y_pred_score
        self._y_pred = self.scorer._y_pred
        self._y_true = self.scorer._y_true

        self._include_text = False

    def metrics(self) -> pd.DataFrame:
        """Returns a DataFrame containing the goodness-of-fit statistics
        for the model.

        Parameters
        ----------
        pd.DataFrame
        """
        return self.scorer.stats_df()


class MNLogitReport:
    """MNLogitReport.
    Fits the model based on provided DataHandler.
    Wraps train and test SingleDatasetMNLogitReport objects.
    """

    def __init__(
        self,
        model: MNLogitLinearModel,
        datahandler: DataHandler,
        target: str,
        predictors: list[str],
        dataemitter: DataEmitter | None = None,
    ):
        """LogitRegressionReport.
        Fits the model based on provided DataHandler.
        Wraps train and test SingleDatasetMNLogitReport objects.

        Parameters
        ----------
        model : MNLogitLinearModel

        datahandler : DataHandler
            The DataHandler object that contains the data.

        target : str
            The name of the dependent variable.

        predictors : list[str]
            The names of the independent variables.
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
        self._train_report = _SingleDatasetMNLogitReport(model, "train")
        self._test_report = _SingleDatasetMNLogitReport(model, "test")

    def step(
        self,
        direction: Literal["both", "backward", "forward"] = "backward",
        criteria: Literal["aic", "bic"] = "aic",
        kept_vars: list[str] | None = None,
        all_vars: list[str] | None = None,
        start_vars: list[str] | None = None,
        max_steps: int = 100,
    ) -> "MNLogitReport":
        """Performs stepwise selection on the model. Returns a new
        MNLogitReport object with the updated model.

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
        MNLogitReport
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
        new_emitter.select_predictors_pre_onehot(selected_vars)

        return MNLogitReport(
            MNLogitLinearModel(),
            self._datahandler,
            self._target,
            self._predictors,
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

    def model(self) -> MNLogitLinearModel:
        """Returns the fitted MNLogitLinearModel object.

        Returns
        -------
        MNLogitLinearModel
        """
        return self._model

    def metrics(self, dataset: Literal["train", "test"]) -> pd.DataFrame:
        """Returns a DataFrame containing the goodness-of-fit statistics
        for the model.

        Parameters
        ----------
        dataset : Literal['train', 'test']
            The dataset to generate the report for.

        Returns
        -------
        pd.DataFrame
        """
        if dataset == "train":
            return self._train_report.metrics()
        elif dataset == "test":
            return self._test_report.metrics()
        else:
            raise ValueError('specification must be either "train" or "test".')
