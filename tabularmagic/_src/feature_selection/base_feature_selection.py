from ..data.datahandler import DataEmitter


class BaseFeatureSelector:
    """A feature selection class."""

    def __init__(self, name: str | None = None):
        """
        Constructs a BaseFeatureSelectorR.

        Parameters
        ----------
        - nickname

        Returns
        -------
        - None
        """
        self._name = name

    def select(self, dataemitter: DataEmitter, n_target_features: int):
        """
        Selects the top n_target_features features
        based on the training data.

        Parameters
        ----------
        - dataemitter : DataEmitter.
        - n_target_features : int.
            Number of desired features, < n_predictors.

        Returns
        -------
        - np array ~ (n_features).
            Selected features.
        - np array ~ (n_features).
            Boolean mask.
        """
        return None, None

    def __str__(self):
        return self._name
