import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal
import warnings
from .base import BaseR
from ....data.datahandler import DataHandler
from ....metrics.visualization import plot_obs_vs_pred
from ....display.print_utils import print_wrapped
from ....feature_selection import BaseFSR, VotingSelectionReport


warnings.simplefilter("ignore", category=UserWarning)


class SingleModelSingleDatasetMLRegReport:
    """
    Class for generating regression-relevant plots and
    tables for a single machine learning model on a single dataset.
    """

    def __init__(self, model: BaseR, dataset: Literal["train", "test"]):
        """
        Initializes a SingleModelSingleDatasetMLReport object.

        Parameters
        ----------
        model : BaseRegression
            The data for the model must already be
            specified. The model should already be trained on the specified data.

        dataset : Literal['train', 'test']
        """
        self._model = model
        if dataset not in ["train", "test"]:
            raise ValueError('dataset must be either "train" or "test".')
        self._dataset = dataset

    def metrics(self) -> pd.DataFrame:
        """Returns a DataFrame containing the goodness-of-fit statistics
        for the model on the specified data.

        Returns
        ----------
        pd.DataFrame
        """
        if self._dataset == "train":
            return self._model._train_scorer.stats_df()
        else:
            return self._model._test_scorer.stats_df()

    def cv_metrics(self, average_across_folds: bool = True) -> pd.DataFrame | None:
        """Returns a DataFrame containing the cross-validated goodness-of-fit
        statistics for the model on the specified data.

        Parameters
        ----------
        average_across_folds : bool
            Default: True. If True, returns a DataFrame
            containing goodness-of-fit statistics averaged across all folds.
            Otherwise, returns a DataFrame containing goodness-of-fit
            statistics for each fold.

        Returns
        ----------
        pd.DataFrame | None
            None is returned if cross validation fit statistics are not available.
        """
        if not self._model.is_cross_validated():
            print_wrapped(
                "Cross validation statistics are not available "
                + "for models that are not cross-validated.",
                type="WARNING",
            )
            return None
        if self._dataset == "train":
            if average_across_folds:
                return self._model._cv_scorer.stats_df()
            else:
                return self._model._cv_scorer.cv_stats_df()
        else:
            print_wrapped(
                "Cross validation statistics are not available for test data.",
                type="WARNING",
            )
            return None

    def plot_obs_vs_pred(
        self, figsize: tuple[float, float] = (5, 5), ax: plt.Axes | None = None
    ) -> plt.Figure:
        """Returns a figure that is a scatter plot of the observed (y-axis) and
        predicted (x-axis) values.

        Parameters
        ----------
        figsize : tuple[float, float]
            Default: (5, 5). The size of the figure.

        ax : plt.Axes | None
            Default: None. The axes on which to plot the figure. If None,
            a new figure is created.

        Returns
        -------
        plt.Figure
        """
        if self._dataset == "train":
            y_pred = self._model._train_scorer._y_pred
            y_true = self._model._train_scorer._y_true
        else:
            y_pred = self._model._test_scorer._y_pred
            y_true = self._model._test_scorer._y_true
        return plot_obs_vs_pred(y_pred, y_true, self._model._name, figsize, ax)


class SingleModelMLRegReport:
    """SingleModelMLRegReport: generates regression-relevant plots and
    tables for a single machine learning model.
    """

    def __init__(self, model: BaseR):
        """
        Initializes a SingleModelMLRegReport object.

        Parameters
        ----------
        model : BaseR
            The data for the model must already be specified.
            The model should already be trained on the specified data.
        """
        self._model = model

    def train_report(self) -> SingleModelSingleDatasetMLRegReport:
        """Returns a SingleModelSingleDatasetMLReport object for the training data.

        Returns
        -------
        SingleModelSingleDatasetMLReport
        """
        return SingleModelSingleDatasetMLRegReport(self._model, "train")

    def test_report(self) -> SingleModelSingleDatasetMLRegReport:
        """Returns a SingleModelSingleDatasetMLReport object for the test data.

        Returns
        -------
        SingleModelSingleDatasetMLReport
        """
        return SingleModelSingleDatasetMLRegReport(self._model, "test")

    def model(self) -> BaseR:
        """Returns the model.

        Returns
        -------
        BaseR
        """
        return self._model

    def plot_obs_vs_pred(
        self,
        dataset: Literal["train", "test"] = "test",
        figsize: tuple[float, float] = (5, 5),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is a scatter plot of the observed (y-axis) and
        predicted (x-axis) values for the specified dataset.

        Parameters
        ----------
        dataset : Literal['train', 'test']
            Default: 'test'.
            The dataset for which to plot the observed vs predicted values.

        figsize : tuple[float, float]
            Default: (5, 5). The size of the figure.

        ax : plt.Axes | None
            Default: None. The axes on which to plot the figure. If None,
            a new figure is created.

        Returns
        -------
        plt.Figure
        """
        if dataset == "train":
            return self.train_report().plot_obs_vs_pred(figsize, ax)
        else:
            return self.test_report().plot_obs_vs_pred(figsize, ax)

    def fs_report(self) -> VotingSelectionReport | None:
        """Returns the feature selection report. If feature selectors were
        specified at the model level or not at all, then this method will return None.

        Returns
        -------
        VotingSelectionReport | None
            None is returned if no feature selectors were specified.
        """
        return self._model.fs_report()

    def feature_importance(self) -> pd.DataFrame | None:
        """Returns the feature importances for the model. If the model does not
        have feature importances, the coefficients are returned instead.
        If the model does not have feature importances or coefficients,
        None is returned.

        Returns
        -------
        pd.DataFrame | None
            None is returned if the model does not have feature importances.
        """
        return self._model.feature_importance()


class MLRegressionReport:
    """Class for reporting model goodness of fit.
    Fits the model based on provided DataHandler.
    """

    def __init__(
        self,
        models: list[BaseR],
        datahandler: DataHandler,
        target: str,
        predictors: list[str],
        feature_selectors: list[BaseFSR] | None = None,
        max_n_features: int | None = None,
        outer_cv: int | None = None,
        outer_cv_seed: int = 42,
        verbose: bool = True,
    ):
        """MLRegressionReport.
        Fits the model based on provided DataHandler.

        Parameters
        ----------
        models : list[BaseR]
            The models will be trained by the MLRegressionReport object.

        datahandler : DataHandler
            The DataHandler object that contains the data.

        target : str
            The name of the target variable.

        predictors : list[str]
            The names of the predictor variables.

        feature_selectors : list[BaseFSR] | None
            Default: None.
            The feature selectors for voting selection. Feature selectors
            can be used to select the most important predictors.

        max_n_features : int | None
            Default: None.
            Maximum number of predictors to utilize. Ignored if feature_selectors
            is None.

        outer_cv : int | None
            Default: None.
            If not None, reports training scores via nested k-fold CV.

        outer_cv_seed : int
            Default: 42. The random seed for the outer cross validation loop.

        verbose : bool
            Default: True. If True, prints progress.
        """
        self._models: list[BaseR] = models

        for model in self._models:
            if not isinstance(model, BaseR):
                raise ValueError(
                    f"Model {model} is not an instance of BaseR. "
                    "All models must be instances of BaseR."
                )

        self._id_to_model = {}
        for model in models:
            if model._name in self._id_to_model:
                raise ValueError(f"Duplicate model name: {model._name}.")
            self._id_to_model[model._name] = model

        self._feature_selection_report = None

        self._y_var = target
        self._X_vars = predictors

        self._emitter = datahandler.train_test_emitter(y_var=target, X_vars=predictors)
        if feature_selectors is not None:
            for feature_selector in feature_selectors:
                if not isinstance(feature_selector, BaseFSR):
                    raise ValueError(
                        f"Feature selector {feature_selector} is not an instance of "
                        "BaseFSR. All feature selectors must be instances of BaseFSR."
                    )

            self._feature_selection_report = VotingSelectionReport(
                selectors=feature_selectors,
                dataemitter=self._emitter,
                max_n_features=max_n_features,
                verbose=verbose,
            )
            self._X_vars = self._feature_selection_report.top_features()
            self._emitter.select_predictors(self._X_vars)

        self._emitters = None

        if outer_cv is not None:
            self._emitters = datahandler.kfold_emitters(
                y_var=target,
                X_vars=predictors,
                n_folds=outer_cv,
                shuffle=True,
                random_state=outer_cv_seed,
            )
            if feature_selectors is not None:
                for emitter in self._emitters:
                    fold_selection_report = VotingSelectionReport(
                        selectors=feature_selectors,
                        dataemitter=emitter,
                        max_n_features=max_n_features,
                        verbose=verbose,
                    )
                    emitter.select_predictors(fold_selection_report.top_features())

        self._verbose = verbose

        for model in self._models:
            if self._verbose:
                print_wrapped(f"Evaluating model {model._name}.", type="UPDATE")
            model.specify_data(
                dataemitter=self._emitter,
                dataemitters=self._emitters,
            )

            model.fit(verbose=self._verbose)

            if (
                model._feature_selection_report is not None
                and self._feature_selection_report is not None
            ):
                if self._verbose:
                    print_wrapped(
                        "Feature selectors were specified for all models as well as "
                        f"for the model {str(model)}. "
                        f"The feature selection report attributed to {str(model)} "
                        "will be for the model-specific feature selectors. "
                        "Note that the feature selectors for all models "
                        "were used to select a subset of the predictors first. "
                        "Then, the model-specific feature selectors were used to "
                        "select a subset of the predictors from the subset selected "
                        "by the feature selectors for all models.",
                        type="WARNING",
                        level="INFO",
                    )

            if model._feature_selection_report is None:
                model._set_voting_selection_report(
                    voting_selection_report=self._feature_selection_report
                )

            if self._verbose:
                print_wrapped(
                    f"Successfully evaluated model {model._name}.", type="UPDATE"
                )

        self._id_to_report = {
            model._name: SingleModelMLRegReport(model) for model in models
        }

    def model_report(self, model_id: str) -> SingleModelMLRegReport:
        """Returns the SingleModelMLRegReport object for the specified model.

        Parameters
        ----------
        model_id : str
            The id of the model.

        Returns
        -------
        SingleModelMLRegReport
        """
        if model_id not in self._id_to_report:
            raise ValueError(f"Model {model_id} not found.")
        return self._id_to_report[model_id]

    def model(self, model_id: str) -> BaseR:
        """Returns the model with the specified id.

        Parameters
        ----------
        model_id : str
            The id of the model.

        Returns
        -------
        BaseR
        """
        if model_id not in self._id_to_model:
            raise ValueError(f"Model {model_id} not found.")
        return self._id_to_model[model_id]

    def metrics(self, dataset: Literal["train", "test"] = "test") -> pd.DataFrame:
        """Returns a DataFrame containing the goodness-of-fit statistics for
        all models on the specified data.

        Parameters
        ----------
        dataset : Literal['train', 'test']
            Default: 'test'.

        Returns
        -------
        pd.DataFrame
        """
        if dataset == "train":
            return pd.concat(
                [
                    report.train_report().metrics()
                    for report in self._id_to_report.values()
                ],
                axis=1,
            )
        elif dataset == "test":
            return pd.concat(
                [
                    report.test_report().metrics()
                    for report in self._id_to_report.values()
                ],
                axis=1,
            )
        else:
            raise ValueError('dataset must be either "train" or "test".')

    def cv_metrics(self, average_across_folds: bool = True) -> pd.DataFrame | None:
        """Returns a DataFrame containing the cross-validated goodness-of-fit
        statistics for all models on the training data. Cross validation must
        have been conducted, otherwise None is returned.

        Parameters
        ----------
        average_across_folds : bool
            Default: True.
            If True, returns a DataFrame containing goodness-of-fit
            statistics averaged across all folds.
            Otherwise, returns a DataFrame containing goodness-of-fit
            statistics for each fold.

        Returns
        -------
        pd.DataFrame | None
            None if cross validation was not conducted.
        """
        if not self._models[0].is_cross_validated():
            print_wrapped(
                "Cross validation statistics are not available "
                + "for models that are not cross-validated.",
                type="WARNING",
            )
            return None
        return pd.concat(
            [
                report.train_report().cv_metrics(average_across_folds)
                for report in self._id_to_report.values()
            ],
            axis=1,
        )

    def fs_report(self) -> VotingSelectionReport | None:
        """Returns the feature selection report. If feature selectors were
        specified at the model level or not at all, then this method will return None.

        To access the feature selection report for a specific model, use
        model_report(<model_id>).feature_selection_report().

        Returns
        -------
        VotingSelectionReport | None
            None if feature selectors were not specified.
        """
        if self._feature_selection_report is None:
            print_wrapped(
                "No feature selection report available.",
                type="WARNING",
            )
        return self._feature_selection_report

    def plot_obs_vs_pred(
        self,
        model_id: str,
        dataset: Literal["train", "test"] = "test",
        figsize: tuple[float, float] = (5, 5),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is a scatter plot of the observed (y-axis) and
        predicted (x-axis) values for the specified model and dataset.

        Parameters
        ----------
        model_id : str
            The id of the model.

        dataset : Literal['train', 'test']
            Default: 'test'.

        figsize : tuple[float, float]
            Default: (5, 5). The size of the figure.

        ax : plt.Axes | None
            Default: None. The axes on which to plot the figure. If None,
            a new figure is created.

        Returns
        -------
        plt.Figure
        """
        return self._id_to_report[model_id].plot_obs_vs_pred(dataset, figsize, ax)

    def feature_importance(self, model_id: str) -> pd.DataFrame | None:
        """Returns the feature importances of the model with the specified id.
        If the model does not have feature importances, the coefficients are returned
        instead. Otherwise, None is returned.

        Parameters
        ----------
        model_id : str
            The id of the model.

        Returns
        -------
        pd.DataFrame | None
            None is returned if the model does not have feature importances
            or coefficients.
        """
        return self._id_to_report[model_id].feature_importance()

    def __getitem__(self, model_id: str) -> SingleModelMLRegReport:
        return self._id_to_report[model_id]
