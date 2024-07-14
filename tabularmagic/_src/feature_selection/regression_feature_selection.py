from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    mutual_info_regression,
    r_regression,
    SelectFromModel,
)
from sklearn.linear_model import Lasso
from typing import Literal
from .base_feature_selection import BaseFSR
from ..data.datahandler import DataEmitter


class KBestSelectorR(BaseFSR):
    """Selects the k best features based on the f_regression, r_regression,
    or mutual info regression score.
    """

    def __init__(
        self,
        scorer: Literal["f_regression", "r_regression", "mutual_info_regression"],
        name: str | None = None,
    ):
        """
        Constructs a KBestSelectorR.

        Parameters
        ----------
        scorer : Literal['f_regression', 'r_regression',
            'mutual_info_regression']
        name : str.
            Default: None. If None, then outputs the class name.
        """
        if name is None:
            name = f"KBestSelector({scorer})"
        super().__init__(name)
        self.scorer = scorer

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
        scorer = None
        if self.scorer == "f_regression":
            scorer = f_regression
        elif self.scorer == "mutual_info_regression":
            scorer = mutual_info_regression
        elif self.scorer == "r_regression":
            scorer = r_regression
        selector = SelectKBest(scorer, k=max_n_features)

        X_train, y_train = dataemitter.emit_train_Xy()
        self._all_features = X_train.columns.to_list()
        selector.fit(X=X_train, y=y_train)

        self._selected_features = selector.get_feature_names_out()
        self._all_feature_scores = selector.scores_
        self._support = selector.get_support()
        self._selected_feature_scores = selector.scores_[self._support]
        return self._all_features, self._selected_features, self._support


class LassoSelectorR(BaseFSR):
    """Selects the (at most) k best features via Lasso regression model-inherent
    feature selection based on the training data.
    """

    def __init__(
        self,
        alpha: float = 0.0,
        name: str | None = None,
    ):
        """
        Constructs a LassoSelectorR.

        Parameters
        ----------
        alpha : float.
            Regularization term weight.
        name : str.
            Default: None. If None, then name is set to default.
        """
        if name is None:
            name = f"LassoSelectorR"
        super().__init__(name)
        self.model = Lasso(alpha=alpha)

    def select(self, dataemitter: DataEmitter, max_n_features: int):
        """
        Selects the (at most) top max_n_features features
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
        X_train, y_train = dataemitter.emit_train_Xy()
        self._all_features = X_train.columns.to_list()

        self.model.fit(X=X_train.to_numpy(), y=y_train.to_numpy())
        selector = SelectFromModel(
            estimator=self.model, prefit=True, max_features=max_n_features
        )
        selector.fit(X=X_train, y=y_train)

        self._selected_features = selector.get_feature_names_out()
        self._support = selector.get_support()
        return self._all_features, self._selected_features, self._support
