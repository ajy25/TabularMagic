import pandas as pd
from typing import Iterable, Literal
from ...data.datahandler import DataHandler
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
        predictors: Iterable[str],
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

        predictors : Iterable[str]
            The names of the independent variables.
        """
        self._model = model
        self._datahandler = datahandler
        self._dataemitter = self._datahandler.train_test_emitter(target, predictors)
        self._model.specify_data(self._dataemitter)
        self._model.fit()
        self._target = target

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
        new_datahandler = DataHandler(
            df_train=self._datahandler.df_train(),
            df_test=self._datahandler.df_test(),
        )
        y_scaler = self._datahandler.scaler(self._target)
        if y_scaler is not None:
            new_datahandler.add_scaler(
                scaler=y_scaler,
                var=self._target,
            )
        return CountRegressionReport(
            CountLinearModel(), new_datahandler, self._target, selected_vars
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
