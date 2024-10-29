from ...data import DataEmitter


class BaseCluster:
    """Base class for clustering models."""

    def __init__(self):
        self._id = "BaseClusterModel"

        self._estimator = None

    def specify_data(self, dataemitter: DataEmitter) -> None:
        """
        Specifies the data emitter for the GMM object.

        Parameters
        ----------
        dataemitter : DataEmitter
            The data emitter to be used.
        """
        self._dataemitter = dataemitter

    def fit(self) -> None:
        """Fits the model to the data."""
        self._train_cluster_labels = self._estimator.fit_predict(
            self._dataemitter.emit_train_X()
        )
        self._test_cluster_labels = self._estimator.predict(
            self._dataemitter.emit_test_X()
        )

    def __str__(self):
        return self._id
