import numpy as np
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    mutual_info_regression,
    r_regression,
    SelectFromModel,
)
from sklearn.linear_model import Lasso, LassoCV
from typing import Literal
from .base_feature_selection import BaseFSR
from ..data.datahandler import DataEmitter


class KBestFSR(BaseFSR):
    """Selects the k best features based on the f_regression, r_regression,
    or mutual info regression score.
    """

    def __init__(
        self,
        scorer: Literal["f_regression", "r_regression", "mutual_info_regression"],
        k: int,
        name: str | None = None,
    ):
        """
        Constructs a KBestFSR.

        Parameters
        ----------
        scorer : Literal['f_regression', 'r_regression',
            'mutual_info_regression']

        k : int
            Number of desired features, < n_predictors.

        name : str | None
            Default: None. If None, then outputs the class name.
        """
        if name is None:
            name = f"KBestFSR({scorer})"
        super().__init__(name)
        self._scorer = scorer
        self._k = k

    def select(
        self, dataemitter: DataEmitter
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Selects the top k features
        based on the training data.

        Parameters
        ----------
        dataemitter : DataEmitter

        Returns
        -------
        np.ndarray ~ (n_in_features)
            All features (variable names).

        np.ndarray ~ (n_out_features)
            Selected features.

        np.ndarray ~ (n_in_features)
            Boolean mask, the support for selected features.
        """
        scorer = None
        if self._scorer == "f_regression":
            scorer = f_regression
        elif self._scorer == "mutual_info_regression":
            scorer = mutual_info_regression
        elif self._scorer == "r_regression":
            scorer = r_regression
        selector = SelectKBest(scorer, k=self._k)

        X_train, y_train = dataemitter.emit_train_Xy()
        self._all_features = X_train.columns.to_numpy()
        selector.fit(X=X_train, y=y_train)

        self._selected_features = selector.get_feature_names_out()
        self._all_feature_scores = selector.scores_
        self._support = selector.get_support()
        self._selected_feature_scores = selector.scores_[self._support]
        return self._all_features, self._selected_features, self._support


class LassoFSR(BaseFSR):
    """Selects the (at most) k best features via Lasso regression model-inherent
    feature selection.
    """

    def __init__(
        self,
        max_n_features: int,
        alpha: float | None = None,
        name: str | None = None,
    ):
        """
        Constructs a LassoFSR.

        Parameters
        ----------
        max_n_features : int
            Number of desired features, < n_predictors.

        alpha : float | None
            Default: None. Regularization term weight. If None,
            then alpha is selected via five-fold cross validation from a default
            grid of candidate alphas.

        name : str | None
            Default: None. If None, then name is set to default.
        """
        if name is None:
            name = "LassoFSR"
        super().__init__(name)
        if alpha is None:
            self._model = LassoCV(cv=5)
        else:
            self._model = Lasso(alpha=alpha)
        self._max_n_features = max_n_features

    def select(
        self, dataemitter: DataEmitter
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Selects the (at most) top max_n_features features
        based on the training data.

        Parameters
        ----------
        dataemitter : DataEmitter

        Returns
        -------
        np.ndarray ~ (n_in_features)
            All features (variable names).

        np.ndarray ~ (n_out_features)
            Selected features.

        np.ndarray ~ (n_in_features)
            Boolean mask, the support for selected features.
        """
        X_train, y_train = dataemitter.emit_train_Xy()
        self._all_features = X_train.columns.to_numpy()

        self._model.fit(X=X_train.to_numpy(), y=y_train.to_numpy())
        selector = SelectFromModel(
            estimator=self._model, prefit=True, max_features=self._max_n_features
        )
        selector.fit(X=X_train, y=y_train)

        self._selected_features = selector.get_feature_names_out()
        self._support = selector.get_support()
        self._all_feature_scores = selector.estimator_.coef_
        return self._all_features, self._selected_features, self._support
