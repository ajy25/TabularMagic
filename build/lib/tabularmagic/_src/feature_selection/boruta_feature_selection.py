from typing import Literal
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np

from .base_feature_selection import BaseFSC, BaseFSR
from .BorutaPy import BorutaPy

from ..data.datahandler import DataEmitter


class BorutaFSR(BaseFSR):
    def __init__(
        self,
        estimator: Literal["tree", "rf", "xgb"] = "rf",
        n_estimators: int = 100,
        name: str | None = None,
    ):
        """
        Constructs a BorutaFSR.

        Parameters
        ----------
        estimator : Literal["tree", "rf", "xgb"]
            Default: "rf". The estimator to use for Boruta. Default
            hyperparameters are used for the estimator.

        n_estimators : int
            Default: 100. The number of estimators to use for Boruta.

        name : str | None
            Default: None. If None, then outputs the default name.
        """
        if name is None:
            name = "BorutaFSR"
        super().__init__(name)

        sk_estimator = None
        if estimator == "tree":
            sk_estimator = DecisionTreeRegressor()
        elif estimator == "rf":
            sk_estimator = RandomForestRegressor()
        elif estimator == "xgb":
            sk_estimator = XGBRegressor()
        else:
            raise ValueError(
                f"estimator must be one of 'tree', 'rf', or 'xgb'. Got: {estimator}"
            )

        self._selector = BorutaPy(estimator=sk_estimator, n_estimators=n_estimators)

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
            The selected features.

        np.ndarray ~ (n_in_features)
            The support of the selected features.

        np.ndarray ~ (n_in_features)
            The ranking of the selected features.
        """
        X_train, y_train = dataemitter.emit_train_Xy()
        self._all_features = X_train.columns.to_numpy()

        self._selector.fit(X=X_train.to_numpy(), y=y_train.to_numpy())
        self._support = self._selector.support_
        self._all_feature_scores = self._selector.ranking_
        self._selected_features = self._all_features[self._support]

        return self._all_features, self._selected_features, self._support


class BorutaFSC(BaseFSC):
    def __init__(
        self,
        estimator: Literal["tree", "rf", "xgb"] = "rf",
        n_estimators: int = 100,
        name: str | None = None,
    ):
        """
        Constructs a BorutaFSC.

        Parameters
        ----------
        estimator : Literal["tree", "rf", "xgb"]
            Default: "rf". The estimator to use for Boruta. Default
            hyperparameters are used for the estimator.

        n_estimators : int
            Default: 100. The number of estimators to use for Boruta.

        name : str | None
            Default: None. If None, then outputs the default name.
        """
        if name is None:
            name = "BorutaFSC"
        super().__init__(name)

        sk_estimator = None
        if estimator == "tree":
            sk_estimator = DecisionTreeClassifier()
        elif estimator == "rf":
            sk_estimator = RandomForestClassifier()
        elif estimator == "xgb":
            sk_estimator = XGBClassifier()
        else:
            raise ValueError(
                f"estimator must be one of 'tree', 'rf', or 'xgb'. Got: {estimator}"
            )

        self._selector = BorutaPy(estimator=sk_estimator, n_estimators=n_estimators)

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
            The selected features.

        np.ndarray ~ (n_in_features)
            The support of the selected features.

        np.ndarray ~ (n_in_features)
            The ranking of the selected features.
        """
        X_train, y_train = dataemitter.emit_train_Xy()
        self._all_features = X_train.columns.to_numpy()

        self._selector.fit(X=X_train.to_numpy(), y=y_train.to_numpy())
        self._support = self._selector.support_
        self._all_feature_scores = self._selector.ranking_
        self._selected_features = self._all_features[self._support]

        return self._all_features, self._selected_features, self._support
