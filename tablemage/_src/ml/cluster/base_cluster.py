import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from typing import Literal
from ...data import DataEmitter


class BaseClust:
    """Base class for clustering models."""

    def __init__(
        self,
    ):
        self._dataemitter = None
        self._name = "BaseClusterModel"
        self._estimator: BaseEstimator = None
        self._train_labels = None
        self._test_labels = None
        self._n_clusters = None

    def specify_data(self, dataemitter: DataEmitter) -> None:
        """
        Specifies the data emitter for the BaseClust object.

        Parameters
        ----------
        dataemitter : DataEmitter
            The data emitter to be used.
        """
        self._dataemitter = dataemitter

    def fit(self):
        """Fits the model to the data."""
        pass

    def train_labels(self) -> pd.Series:
        """Returns the training labels."""
        return self._train_labels

    def test_labels(self) -> pd.Series:
        """Returns the test labels."""
        return self._test_labels

    def labels(self, dataset: Literal["train", "test"]) -> pd.Series:
        """Returns the labels for the specified dataset.

        Parameters
        ----------
        dataset : Literal["train", "test"]
            Dataset to obtain labels from.

        Returns
        -------
        pd.Series
            Labels for the specified dataset.
        """
        if dataset == "train":
            return self.train_labels()
        elif dataset == "test":
            return self.test_labels()
        else:
            raise ValueError("dataset must be 'train' or 'test'.")

    def sklearn_model(self) -> BaseEstimator:
        """Returns the best estimator (sklearn estimator object). The best estimator
        was fitted on the train data through the hyperparameter search process.

        Note that the sklearn estimator can be saved and used for future predictions.
        However, the input data must be preprocessed in the same way. If you intend
        to use the estimator for future predictions, it is recommended that you
        manually specify every preprocessing step, which will ensure that you
        have full control over how the data is being transformed for future
        reproducibility and predictions.

        It is not recommended to use TableMage for ML production.
        We recommend using TableMage to quickly identify promising models
        and then manually implementing and training
        the best model in a production environment.

        Returns
        -------
        BaseEstimator"""
        return self._estimator

    def sklearn_pipeline(self) -> Pipeline:
        """Returns an sklearn pipeline object. The pipeline allows for
        retrieving model predictions directly from data formatted like the original
        train and test data.

        It is not recommended to use TableMage for ML production.
        We recommend using TableMage to quickly identify promising models
        and then manually implementing and training
        the best model in a production environment.

        Returns
        -------
        Pipeline | InverseTransformRegressor
            Returns either a Pipeline (from scikit-learn).
        """
        pipeline = Pipeline(
            steps=[
                (
                    "custom_prep_data",
                    self._dataemitter.sklearn_preprocessing_transformer(),
                ),
                ("model", self._estimator),
            ]
        )
        return pipeline

    def n_clusters(self) -> int:
        """Returns the number of clusters."""
        return self._n_clusters

    def __str__(self):
        return self._name
