import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal, Any
import warnings
from .base import BaseC
from ....metrics.classification_scoring import ClassificationBinaryScorer
from ....data.datahandler import DataHandler
from ....metrics.visualization import plot_roc_curve, plot_confusion_matrix
from ....display.print_utils import print_wrapped
from ....feature_selection import BaseFSC, VotingSelectionReport


warnings.simplefilter("ignore", category=UserWarning)


class SingleModelSingleDatasetMLClassReport:
    """
    Class for generating classification-relevant plots and
    tables for a single machine learning model on a single dataset.
    """

    def __init__(self, model: BaseC, dataset: Literal["train", "test"]):
        """
        Initializes a SingleModelSingleDatasetMLClassReport object.

        Parameters
        ----------
        model: BaseC
            The data for the model must already be
            specified. The model should already be trained on the
            specified data.

        dataset: Literal['train', 'test']
            The dataset to generate the report for.
        """
        self._model = model
        self._is_binary = isinstance(model._train_scorer, ClassificationBinaryScorer)
        if dataset not in ["train", "test"]:
            raise ValueError('dataset must be either "train" or "test".')
        self._dataset = dataset

    def metrics(self) -> pd.DataFrame:
        """Returns a DataFrame containing the evaluation metrics
        for the model on the specified data.

        Returns
        -------
        pd.DataFrame
        """
        if self._dataset == "train":
            return self._model._train_scorer.stats_df()
        else:
            return self._model._test_scorer.stats_df()

    def metrics_by_class(self) -> pd.DataFrame | None:
        """Returns a DataFrame containing the evaluation metrics
        for the model on the specified data, broken down by class.

        Returns
        -------
        pd.DataFrame | None
            None is returned if the model is binary.
        """
        if self._is_binary:
            print_wrapped(
                "Fit statistics by class are not "
                + "available for binary classification.",
                type="WARNING",
            )
            return None

        if self._dataset == "train":
            return self._model._train_scorer.stats_by_class_df()
        else:
            return self._model._test_scorer.stats_by_class_df()

    def cv_metrics(self, average_across_folds: bool = True) -> pd.DataFrame | None:
        """Returns a DataFrame containing the cross-validated evaluation metrics
        for the model on the specified data.

        Parameters
        ----------
        average_across_folds : bool
            Default: True. If True, returns a DataFrame
            containing goodness-of-fit statistics across all folds.

        Returns
        -------
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
        elif self._dataset == "test":
            print_wrapped(
                "Cross validation statistics are not available for test data.",
                type="WARNING",
            )
            return None

    def cv_metrics_by_class(
        self, averaged_across_folds: bool = True
    ) -> pd.DataFrame | None:
        """Returns a DataFrame containing the cross-validated evaluation metrics
        for the model on the specified data, broken down by class.

        Parameters
        ----------
        averaged_across_folds : bool
            Default: True. If True, returns a DataFrame
            containing goodness-of-fit statistics across all folds.

        Returns
        -------
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

        if self._is_binary:
            print_wrapped(
                "Cross validation statistics by class are not "
                + "available for binary classification.",
                type="WARNING",
            )
            return None

        if self._dataset == "train":
            if averaged_across_folds:
                return self._model._cv_scorer.stats_by_class_df()
            else:
                return self._model._cv_scorer.cv_stats_by_class_df()
        else:
            print_wrapped(
                "Cross validation statistics are not available for test data.",
                type="WARNING",
            )
            return None

    def plot_confusion_matrix(
        self, figsize: tuple[float, float] = (5, 5), ax: plt.Axes | None = None
    ) -> plt.Figure:
        """Returns a figure that is the confusion matrix for the model.

        Parameters
        ----------
        figsize: tuple[float, float]
            Default: (5, 5). The size of the figure.

        ax: plt.Axes
            Default: None. The axes on which to plot the figure. If None,
            a new figure is created.

        Returns
        -------
        plt.Figure
            Figure of the confusion matrix.
        """
        if self._dataset == "train":
            y_pred = self._model._train_scorer._y_pred
            y_true = self._model._train_scorer._y_true
        else:
            y_pred = self._model._test_scorer._y_pred
            y_true = self._model._test_scorer._y_true
        return plot_confusion_matrix(y_pred, y_true, self._model._name, figsize, ax)

    def plot_roc_curve(
        self,
        label_curve: bool = False,
        color: str | Any = "black",
        figsize: tuple[float, float] = (5, 5),
        ax: plt.Axes | None = None,
    ) -> plt.Figure | None:
        """Returns a figure that is the ROC curve for the model.

        Parameters
        ----------
        label_curve : bool
            Default: False. Whether to label the ROC curve with model name and AUC.
            If True, the model name and AUC are displayed on the ROC curve rather
            than in the title. This is useful when plotting multiple ROC curves
            on the same axes.

        color : str | Any
            Default: "black". The color of the ROC curve.

        figsize: tuple[float, float]
            Default: (5, 5). The size of the figure.

        ax: plt.Axes | None
            Default: None. The axes on which to plot the figure. If None,
            a new figure is created.

        Returns
        -------
        plt.Figure | None
            Figure of the ROC curve. None is returned if the model is not binary.
        """
        if not self._is_binary:
            print_wrapped(
                "ROC curve is not available for " + "multiclass classification.",
                type="WARNING",
            )
            return None

        if self._dataset == "train":
            y_score = self._model._train_scorer._y_pred_score
            y_true = self._model._train_scorer._y_true
        else:
            y_score = self._model._test_scorer._y_pred_score
            y_true = self._model._test_scorer._y_true
        return plot_roc_curve(
            y_score=y_score,
            y_true=y_true,
            model_name=self._model._name,
            label_curve=label_curve,
            color=color,
            figsize=figsize,
            ax=ax,
        )


class SingleModelMLClassReport:
    """Class for routing to appropriate
    SingleModelSingleDatasetMLClassReport object.
    """

    def __init__(self, model: BaseC):
        """
        Initializes a SingleModelMLClassReport object.

        Parameters
        ----------
        model: BaseC
            The data for the model must already be
            specified. The model should already be trained on the
            specified data.
        """
        self._model = model

    def train_report(self) -> SingleModelSingleDatasetMLClassReport:
        """Returns a SingleModelSingleDatasetMLClassReport
            object for the training data.

        Returns
        -------
        SingleModelSingleDatasetMLClassReport
        """
        return SingleModelSingleDatasetMLClassReport(self._model, "train")

    def test_report(self) -> SingleModelSingleDatasetMLClassReport:
        """Returns a SingleModelSingleDatasetMLClassReport
          object for the test data.

        Returns
        -------
        SingleModelSingleDatasetMLClassReport
        """
        return SingleModelSingleDatasetMLClassReport(self._model, "test")

    def plot_confusion_matrix(
        self,
        dataset: Literal["train", "test"] = "test",
        figsize: tuple[float, float] = (5, 5),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is the confusion matrix for the model.

        Parameters
        ----------
        dataset: Literal['train', 'test']
            Default: 'test'. The dataset to plot the confusion matrix for.

        figsize: tuple[float, float]
            Default: (5, 5). The size of the figure.

        ax: plt.Axes | None
            Default: None. The axes on which to plot the figure. If None,
            a new figure is created.

        Returns
        -------
        plt.Figure
            Figure of the confusion matrix.
        """
        if dataset == "train":
            return self.train_report().plot_confusion_matrix(figsize, ax)
        elif dataset == "test":
            return self.test_report().plot_confusion_matrix(figsize, ax)
        else:
            raise ValueError('dataset must be either "train" or "test".')

    def plot_roc_curve(
        self,
        dataset: Literal["train", "test"] = "test",
        figsize: tuple[float, float] = (5, 5),
        ax: plt.Axes | None = None,
    ) -> plt.Figure | None:
        """Returns a figure that is the ROC curve for the model.

        Parameters
        ----------
        dataset: Literal['train', 'test']
            Default: 'test'. The dataset to plot the ROC curve for.

        figsize: tuple[float, float]
            Default: (5, 5). The size of the figure.

        ax: plt.Axes | None
            Default: None. The axes on which to plot the figure. If None,
            a new figure is created.

        Returns
        -------
        plt.Figure | None
            Figure of the ROC curve. None is returned if the model is not binary.
        """
        if dataset == "train":
            return self.train_report().plot_roc_curve(figsize=figsize, ax=ax)
        else:
            return self.test_report().plot_roc_curve(figsize=figsize, ax=ax)

    def model(self) -> BaseC:
        """Returns the model.

        Returns
        -------
        BaseC.
        """
        return self._model

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


class MLClassificationReport:
    """Class for evaluating multiple classification models.
    Fits the model based on provided DataHandler.
    """

    def __init__(
        self,
        models: list[BaseC],
        datahandler: DataHandler,
        target: str,
        predictors: list[str],
        feature_selectors: list[BaseFSC] | None = None,
        max_n_features: int | None = None,
        outer_cv: int | None = None,
        outer_cv_seed: int = 42,
        verbose: bool = True,
    ):
        """MLClassificationReport.
        Fits the model based on provided DataHandler.

        Parameters
        ----------
        models: list[BaseC]
            The models will be trained by the MLClassificationReport.

        datahandler: DataHandler
            The DataHandler object that contains the data.

        target : str
            The name of the dependent variable.

        predictors : list[str]
            The names of the independent variables.

        feature_selectors : list[BaseFSR] | None
            Default: None.
            The feature selectors for voting selection. Feature selectors
            can be used to select the most important predictors.

        max_n_features : int | None
            Default: None.
            Maximum number of predictors to utilize. Ignored if feature_selectors
            is None.

        outer_cv: int | None
            Default: None.
            If not None, reports training scores via nested k-fold CV.

        outer_cv_seed: int
            Default: 42.
            The random seed for the outer cross validation loop.

        verbose: bool
            Default: True. If True, prints updates on model fitting.
        """
        self._models: list[BaseC] = models

        for model in self._models:
            if not isinstance(model, BaseC):
                raise ValueError(
                    f"Model {model} is not an instance of BaseC. "
                    "All models must be instances of BaseC."
                )

        self._id_to_model = {}
        for model in models:
            if model._name in self._id_to_model:
                raise ValueError(f"Duplicate model name: {model._name}.")
            self._id_to_model[model._name] = model

        self._y_var = target
        self._X_vars = predictors
        self._feature_selection_report = None

        self._emitter = datahandler.train_test_emitter(y_var=target, X_vars=predictors)
        if feature_selectors is not None:
            for feature_selector in feature_selectors:
                if not isinstance(feature_selector, BaseFSC):
                    raise ValueError(
                        f"Feature selector {feature_selector} is not an instance of "
                        "BaseFSC. All feature selectors must be instances of BaseFSC."
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
            model.specify_data(dataemitter=self._emitter, dataemitters=self._emitters)
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
            model._name: SingleModelMLClassReport(model) for model in models
        }

    def _model_report(self, model_id: str) -> SingleModelMLClassReport:
        """Returns the SingleModelMLClassReport object for the specified model.

        Parameters
        ----------
        model_id: str
            The id of the model.

        Returns
        -------
        SingleModelMLClassReport
        """
        if model_id not in self._id_to_report:
            raise ValueError(f"Model {model_id} not found.")
        return self._id_to_report[model_id]

    def model(self, model_id: str) -> BaseC:
        """Returns the model with the specified id.

        Parameters
        ----------
        model_id: str
            The id of the model.

        Returns
        -------
        BaseC
        """
        if model_id not in self._id_to_model:
            raise ValueError(f"Model {model_id} not found.")
        return self._id_to_model[model_id]

    def metrics(self, dataset: Literal["train", "test"]) -> pd.DataFrame:
        """Returns a DataFrame containing the evaluation metrics for
        all models on the specified data.

        Parameters
        ----------
        dataset: Literal['train', 'test']
            The dataset to return the fit statistics for.

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
        """Returns a DataFrame containing the evaluation metrics for
        all models on the training data. Cross validation must
        have been conducted, otherwise None is returned.

        Parameters
        ----------
        average_across_folds : bool
            Default: True.
            If True, returns a DataFrame
            containing goodness-of-fit statistics across all folds.

        Returns
        -------
        pd.DataFrame | None
            None is returned if cross validation was not conducted.
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
            None is returned if no feature selectors were specified.
        """
        if self._feature_selection_report is None:
            print_wrapped(
                "No feature selection report available.",
                type="WARNING",
            )
        return self._feature_selection_report

    def plot_confusion_matrix(
        self,
        model_id: str,
        dataset: Literal["train", "test"],
        figsize: tuple[float, float] = (5, 5),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is the confusion matrix for the model.

        Parameters
        ----------
        model_id: str
            The id of the model.

        dataset: Literal['train', 'test']
            The dataset to plot the confusion matrix for.

        figsize: tuple[float, float]
            Default: (5, 5). The size of the figure.

        ax: plt.Axes | None
            Default: None. The axes on which to plot the figure. If None,
            a new figure is created.

        Returns
        -------
        plt.Figure
            Figure of the confusion matrix.
        """
        return self._id_to_report[model_id].plot_confusion_matrix(dataset, figsize, ax)

    def plot_roc_curves(
        self,
        dataset: Literal["train", "test"],
        figsize: tuple[float, float] = (5, 5),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Plots the ROC curves for all models.

        Parameters
        ----------
        dataset: Literal['train', 'test']
            The dataset to plot the ROC curves for.

        figsize: tuple[float, float]
            Default: (5, 5). The size of the figure.

        ax: plt.Axes | None
            Default: None. The axes to plot on. If None, a new figure is created.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.get_figure()

        color_palette = sns.color_palette("tab10")

        for i, model_id in enumerate(self._id_to_report):
            if dataset == "train":
                self._id_to_report[model_id].train_report().plot_roc_curve(
                    label_curve=True, color=color_palette[i], figsize=figsize, ax=ax
                )

            elif dataset == "test":
                self._id_to_report[model_id].test_report().plot_roc_curve(
                    label_curve=True, color=color_palette[i], figsize=figsize, ax=ax
                )

            else:
                raise ValueError('dataset must be either "train" or "test".')

        ax.set_title("ROC Curves")
        ax.legend(
            fontsize=8,  # Set font size directly
            handlelength=1,  # Reduce length of legend handles
            handletextpad=0.5,  # Reduce space between handle and text
            borderpad=0.2,  # Reduce internal padding
            labelspacing=0.2,  # Reduce vertical space between legend entries
            loc="lower right",  # Set a specific location
        )
        fig.tight_layout()
        plt.close(fig)

        return fig

    def plot_roc_curve(
        self,
        model_id: str,
        dataset: Literal["train", "test"],
        figsize: tuple[float, float] = (5, 5),
        ax: plt.Axes | None = None,
    ) -> plt.Figure | None:
        """Plots the ROC curve for a single model.

        Parameters
        ----------
        model_id: str
            The id of the model.

        dataset: Literal['train', 'test']
            The dataset to plot the ROC curve for.

        figsize: tuple[float, float]
            Default: (5, 5). The size of the figure.

        Returns
        -------
        plt.Figure | None
            Figure of the ROC curve. None is returned if the model is not binary.
        """
        return self._id_to_report[model_id].plot_roc_curve(dataset, figsize, ax)

    def metrics_by_class(
        self, dataset: Literal["train", "test"]
    ) -> pd.DataFrame | None:
        """Returns a DataFrame containing the evaluation metrics
        for all models on the specified data, broken down by class.

        Parameters
        ----------
        dataset: Literal['train', 'test']
            The dataset to return the fit statistics for.

        Returns
        -------
        pd.DataFrame | None
            None is returned if the model is binary.
        """
        if self._models[0].is_binary():
            print_wrapped(
                "Fit statistics by class are not "
                + "available for binary classification.",
                type="WARNING",
            )
            return
        if dataset == "train":
            return pd.concat(
                [
                    report.train_report().metrics_by_class()
                    for report in self._id_to_report.values()
                ],
                axis=1,
            )
        elif dataset == "test":
            return pd.concat(
                [
                    report.test_report().metrics_by_class()
                    for report in self._id_to_report.values()
                ],
                axis=1,
            )
        else:
            raise ValueError('dataset must be either "train" or "test".')

    def cv_metrics_by_class(
        self,
        averaged_across_folds: bool = True,
    ) -> pd.DataFrame | None:
        """Returns a DataFrame containing the cross-validated evaluation metrics
        for all models on the specified data, broken down by class.

        Parameters
        ----------
        averaged_across_folds : bool
            Default: True. If True, returns a DataFrame
            containing goodness-of-fit statistics across all folds.

        Returns
        -------
        pd.DataFrame | None
            None is returned if cross validation was not conducted.
        """
        if not self._models[0].is_cross_validated():
            print_wrapped(
                "Cross validation statistics are not available "
                + "for models that are not cross-validated.",
                type="WARNING",
            )
            return None
        if self._models[0].is_binary():
            print_wrapped(
                "Cross validation statistics by class are not "
                + "available for binary classification.",
                type="WARNING",
            )
            return None
        return pd.concat(
            [
                report.train_report().cv_metrics_by_class(averaged_across_folds)
                for report in self._id_to_report.values()
            ],
            axis=1,
        )

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

    def __getitem__(self, model_id: str) -> SingleModelMLClassReport:
        return self._id_to_report[model_id]
