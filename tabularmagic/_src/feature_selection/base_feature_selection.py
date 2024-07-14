from ..data.datahandler import DataEmitter


class BaseFS:
    """A basic feature selection class."""

    def __init__(self, name: str | None = None):
        """
        Constructs a BaseFS.

        Parameters
        ----------
        - nickname

        Returns
        -------
        - None
        """
        self._name = name
        self._all_features, self._selected_features, self._support = None, None, None

    def select(self, dataemitter: DataEmitter, max_n_features: int):
        """
        Selects the top max_n_features features
        based on the training data.

        Parameters
        ----------
        dataemitter : DataEmitter.
        max_n_features : int.
            Number of desired features, < n_predictors.

        Returns
        -------
        np.ndarray ~ (n_in_features).
            All features (variable names).
        np.ndarray ~ (n_out_features).
            Selected features.
        np.ndarray ~ (n_in_features).
            Boolean mask, the support for selected features.
        """
        return None, None, None

    def all_features(self):
        """
        Returns all the considered features.

        Returns
        -------
        np.ndarray ~ (n_in_features)
        """
        return self._all_features

    def selected_features(self):
        """
        Returns the selected features.

        Returns
        -------
        np.ndarray ~ (n_out_features)
        """
        return self._selected_features

    def support(self):
        """
        Returns the support for the selected features.

        Returns
        -------
        np.ndarray ~ (n_in_features)
        """
        return self._support

    def __str__(self):
        return self._name


class BaseFSR(BaseFS):
    """A basic feature selection class for regression tasks."""

    def __init__(self, name: str | None = None):
        """
        Constructs a BaseFSR.

        Parameters
        ----------
        - nickname

        Returns
        -------
        - None
        """
        if name is None:
            name = "BaseFSR"
        super().__init__(name)


class BaseFSC(BaseFS):
    """A basic feature selection class for classification tasks."""

    def __init__(self, name: str | None = None):
        """
        Constructs a BaseFSC.

        Parameters
        ----------
        - nickname

        Returns
        -------
        - None
        """
        if name is None:
            name = "BaseFSC"
        super().__init__(name)
