import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Iterable, Literal
from ...data.datahandler import DataHandler
from ...exploratory.visualization import plot_obs_vs_pred, decrease_font_sizes_axs

from .negbinglmreg import SingleDatasetNegBinRegReport
from .poissonglmreg import SingleDatasetPoisRegReport
from ..countglm import CountLinearModel
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
        y_var: str,
        X_vars: Iterable[str],
    ):
        """CountRegressionReport.
        Fits the model based on provided DataHandler.
        Wraps train and test SingleDatasetPoisRegReport or
        SingleDatasetNegBinRegReport objects depending on chosen model.

        Parameters
        ----------
        model : CountLinearModel.
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

        if self._model._type == "poisson":
            self._train_report = SingleDatasetPoisRegReport(model, "train")
            self._test_report = SingleDatasetPoisRegReport(model, "test")
        else:
            self._train_report = SingleDatasetNegBinRegReport(model, "train")
            self._test_report = SingleDatasetNegBinRegReport(model, "test")

    def train_report(self) -> SingleDatasetPoisRegReport | SingleDatasetNegBinRegReport:
        """Returns an CountRegressionReport object for the train dataset

        Returns
        -------
        report : CountRegressionReport.
        """
        return self._train_report

    def test_report(self) -> SingleDatasetPoisRegReport | SingleDatasetNegBinRegReport:
        """Returns an CountRegressionReport object for the test dataset

        Returns
        -------
        report : CountRegressionReport.
        """
        return self._test_report

    def model(self) -> CountLinearModel:
        """Returns the fitted CountLinearModel
        object.

        Returns
        -------
        CountLinearModel.
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

    def stepwise(self) -> "CountRegressionReport":
        """Performs stepwise selection on the model.

        Parameters
        ----------
        alpha : float.
            Default is 0.05.

        Returns
        -------
        CountLinearModel.
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
