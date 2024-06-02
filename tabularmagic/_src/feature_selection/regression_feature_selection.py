from sklearn.feature_selection import (SelectKBest, f_regression, 
    mutual_info_regression, r_regression, RFE, SelectFromModel, 
    SequentialFeatureSelector)
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
import pandas as pd
from typing import Literal

from ..data.datahandler import DataEmitter



class BaseFeatureSelectorR():
    """A feature selection class.
    """

    def __init__(self, name: str = None):
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


    def select(self, 
               dataemitter: DataEmitter,
               n_target_features: int):
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





class KBestSelectorR(BaseFeatureSelectorR):
    """Selects the k best features based on the f_regression or mutual info 
    regression score.
    """

    def __init__(self, scorer: Literal['f_regression', 'r_regression',
            'mutual_info_regression'], name: str = None):
        """
        Constructs a KBestSelectorR.

        Parameters
        ----------
        - scorer : Literal['f_regression', 'r_regression', 
            'mutual_info_regression']
        - nickname : str.
            Default: None. If None, then outputs the class name. 

        Returns
        -------
        - None
        """
        super().__init__(name)
        if self._name is None:
            self._name = f'KBestSelector({scorer})'
        self.scorer = scorer


    def select(self, 
               dataemitter: DataEmitter,
               n_target_features: int):
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
        scorer = None
        if self.scorer == 'f_regression':
            scorer = f_regression
        elif self.scorer == 'mutual_info_regression':
            scorer = mutual_info_regression
        elif self.scorer == 'r_regression':
            scorer = r_regression
        selector = SelectKBest(scorer, k=n_target_features)

        X_train, y_train = dataemitter.emit_train_Xy()

        selector.fit(
            X=X_train,
            y=y_train
        )

        self.selected_features = selector.get_feature_names_out()
        self.all_feature_scores = selector.scores_
        self.support = selector.get_support()
        self.selected_feature_scores = selector.scores_[self.support]
        return self.selected_features, self.support




class SimpleLinearSelectorR(BaseFeatureSelectorR):
    """Selects the (at most) k best features via Lasso regression model-inherent 
    feature selection based on the training data.
    """

    def __init__(self, regularization_type: Literal[None, 'l1', 'l2'] = None, 
                 alpha = 0.0, name: str = None):
        """
        Constructs a SimpleLinearSelectorR.

        Parameters
        ----------
        - regularization_type: Literal[None, 'l1', 'l2'].
        - alpha : float.
            Regularization term weight.
        - nickname : str.
            Default: None. If None, then outputs the class name. 

        Returns
        -------
        - None
        """
        super().__init__(name)
        if self._name is None:
            self._name = f'LinearSelector({regularization_type}, {alpha})'
        if regularization_type == 'l1':
            self.model = Lasso(alpha=alpha)
        elif regularization_type == 'l2':
            self.model = Ridge(alpha=alpha)
        elif regularization_type == None:
            self.model = LinearRegression()
        else:
            raise ValueError(f'Invalid input: regularization_type = ' + \
                             f'{regularization_type}')

    def select(self, 
               dataemitter: DataEmitter,
               n_target_features: int):
        """
        Selects the (at most) top n_target_features features
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
        X_train, y_train = dataemitter.emit_train_Xy()

        self.model.fit(
            X=X_train.to_numpy(),
            y=y_train.to_numpy()
        )
        selector = SelectFromModel(estimator=self.model, prefit=True, 
            max_features=n_target_features)
        selector.fit(
            X=X_train,
            y=y_train
        )

        self.selected_features = selector.get_feature_names_out()
        self.support = selector.get_support()
        return self.selected_features, self.support
    


class RFESelectorR(BaseFeatureSelectorR):
    """Selects the k best features via L1-based 
    recursive feature elimination based on the training data.
    """

    def __init__(self, 
                 model: Literal['ols', 'l1', 'l2', 'decision_tree', 
                    'svm', 'random_forest'] | BaseEstimator, 
                 name: str = None):
        """
        Constructs a RFESelectorR. 

        Parameters
        ----------
        - model : str | BaseEstimator.
        - nickname : str.
            Default: None. If None, then outputs the class name. 

        Returns
        -------
        - None
        """
        super().__init__(name)
        if self._name is None:
            self._name = f"RFESelector({model})"
        if isinstance(model, str):
            if model == 'ols':
                self.model = LinearRegression()
            elif model == 'l1':
                self.model = Lasso()
            elif model == 'l2':
                self.model = Ridge()
            elif model == 'decision_tree':
                self.model = DecisionTreeRegressor()
            elif model == 'random_forest':
                self.model = RandomForestRegressor()
            elif model == 'svm':
                self.model = SVR()
        elif isinstance(model, BaseEstimator):
            self.model = model
        else:
            raise ValueError('Invalid input: model.')


    def select(self, 
               dataemitter: DataEmitter,
               n_target_features: int):
        """
        Selects the (at most) top n_target_features features
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
        X_train, y_train = dataemitter.emit_train_Xy()

        selector = RFE(estimator=self.model, 
                       n_features_to_select=n_target_features)
        selector.fit(
            X=X_train,
            y=y_train
        )
        self.selected_features = selector.get_feature_names_out()
        self.support = selector.get_support()
        return self.selected_features, self.support


