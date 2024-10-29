from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from ..base_cluster import BaseCluster
from ....data import DataEmitter


class GMMCluster(BaseCluster):
    """Gaussian Mixture Model (GMM) clustering.

    GMM is a probabilistic model that assumes that the data is generated
    from a mixture of several Gaussian distributions with unknown parameters.
    The model is trained using the Expectation-Maximization (EM) algorithm.
    """

    def __init__(self, k: int = 1, model_random_state: int = 42):
        """
        Initializes a GMM object.

        Parameters
        ----------
        k : int
            Default: 1. The number of components in the mixture model.

        model_random_state : int
            Default: 42. The random seed to use for reproducibility.
        """
        super().__init__()
        self._id = "GMM"
        self._n_components = k
        self._random_state = model_random_state
        self._estimator = GaussianMixture(
            n_components=k, random_state=model_random_state
        )
        self._train_cluster_labels = None
        self._test_cluster_labels = None

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
        """Fits the GMM model to the data."""
        self._train_cluster_labels = self._estimator.fit_predict(
            self._dataemitter.emit_train_X()
        )
        self._test_cluster_labels = self._estimator.predict(
            self._dataemitter.emit_test_X()
        )

    def sklearn_estimator(self) -> GaussianMixture:
        """Returns the GMM model."""
        return self._estimator

    def sklearn_pipeline(self) -> Pipeline:
        """Returns the GMM model as a pipeline."""

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
