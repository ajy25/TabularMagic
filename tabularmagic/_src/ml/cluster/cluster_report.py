from .base_cluster import BaseCluster
from typing import Literal
from ...data.datahandler import DataHandler


class ClusterReport:
    """Class for reporting clustering results.
    Fits models based on provided DataHandler.
    """

    def __init__(
        self,
        models: list[BaseCluster],
        datahandler: DataHandler,
        features: list[str],
        n_clusters: int | None,
        dataset: Literal["train", "all"],
    ):
        """Initializes ClusterReport.

        Parameters
        ----------
        models : list[BaseCluster]
            List of models to fit.

        datahandler : DataHandler
            DataHandler object.

        features : list[str]
            List of feature names.

        n_clusters : int | None
            Number of clusters to fit.

        dataset : Literal["train", "all"]
            Dataset to fit models on.
            If "train", only fits models on training data.
            Then, predictions can be made on test data.
            If "all", fits models on all data.
        """
        self._models = models

        for model in self._models:
            if not isinstance(model, BaseCluster):
                raise TypeError("All models must be of type BaseCluster.")

        self._X_vars = features

        if dataset == "train":
            self._emitter = datahandler.train_test_emitter(
                y_var=None,
                X_vars=self._X_vars,
            )

        elif dataset == "all":
            self._emitter = datahandler.full_dataset_emitter(
                y_var=None,
                X_vars=self._X_vars,
            )

        else:
            raise ValueError("dataset must be 'train' or 'all'.")

        self._n_cluster = n_clusters

        for model in self._models:
            model.specify_data(self._emitter)
            model.fit()
