from sklearn.feature_selection import (
    SelectKBest,
    SelectFromModel,
    f_classif,
    mutual_info_classif,
    chi2,
)
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from typing import Literal
import numpy as np

from ..data.datahandler import DataEmitter
from .base_feature_selection import BaseFSC


class KBestFSC(BaseFSC):
    """Selects the k best features based on the f_classif or mutual info
    regression score.
    """

    def __init__(
        self,
        scorer: Literal["f_classif", "mutual_info_classif", "chi2"],
        k: int,
        name: str | None = None,
    ):
        """Initializes a KBestFSC object.

        Parameters
        ----------
        scorer : Literal['f_classif', 'mutual_info_classif']

        k : int
            Number of desired features, < n_predictors.

        name : str | None
            Default: None. If None, then outputs the default name.
        """
        if name is None:
            name = f"KBestFSC({scorer})"
        super().__init__(name)
        self._scorer = scorer
        self._k = k

    def select(
        self, dataemitter: DataEmitter
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Selects the top max_n_features features based on the training data.

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
        if self._scorer == "f_classif":
            scorer = f_classif
        elif self._scorer == "mutual_info_classif":
            scorer = mutual_info_classif
        elif self._scorer == "chi2":
            scorer = chi2
        else:
            raise ValueError(f"Invalid scorer: {self._scorer}")
        selector = SelectKBest(scorer, k=self._k)

        X_train, y_train = dataemitter.emit_train_Xy()
        self._all_features = X_train.columns.to_list()
        selector.fit(X=X_train, y=y_train)

        self._selected_features = selector.get_feature_names_out()
        self._all_feature_scores = selector.scores_
        self._support = selector.get_support()
        self._selected_feature_scores = selector.scores_[self._support]
        return self._all_features, self._selected_features, self._support


class LassoFSC(BaseFSC):
    """Selects the (at most) k best features via Lasso regression model-inherent
    feature selection."""

    def __init__(
        self,
        max_n_features: int,
        c: float | None = None,
        name: str | None = None,
    ):
        """
        Constructs a LassoFSC.

        Parameters
        ----------
        max_n_features : int
            Number of desired features, < n_predictors.

        c : float | None
            Default: None. Inverse of regularization strength. If None,
            then c is selected via five-fold cross validation from a grid of 10
            candidate values, on a log scale from 1e-4 to 1e4.

        name : str | None
            Default: None. If None, then name is set to default.
        """

        if name is None:
            name = "LassoFSC"
        super().__init__(name)
        if c is None:
            self._model = LogisticRegressionCV(penalty="l1", solver="liblinear", cv=5)
        else:
            self._model = LogisticRegression(penalty="l1", solver="liblinear", C=c)
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
