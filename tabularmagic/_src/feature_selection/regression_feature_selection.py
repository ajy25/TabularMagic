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


class RegressionBaseSelector():
    """A feature selection class.
    """

    def __init__(self, nickname: str = None):
        """
        Constructs a RegressionBaseSelector.

        Parameters
        ----------
        - nickname

        Returns
        -------
        - None
        """
        self.nickname = nickname


    def select(self, df: pd.DataFrame, X_vars: list[str], y_var: str, 
               n_target_features: int):
        """
        Selects the top n_target_features features.

        Parameters
        ----------
        - df : pd.DataFrame.
        - X_vars : list[str].
            A list of features to look through. 
        - y_var : str.
            The variable to be predicted.
        - n_target_features : int. 
            Number of desired features, < len(X_vars).

        Returns
        -------
        - np array ~ (n_features).
            Selected features.
        - np array ~ (n_features).
            Boolean mask.
        """
        return None, None


    def __str__(self):
        return self.nickname





class KBestSelectorR(RegressionBaseSelector):
    """Selects the k best features based on the f_regression or mutual info 
    regression score.
    """

    def __init__(self, scorer: Literal['f_regression', 'r_regression',
            'mutual_info_regression'], nickname: str = None):
        """
        Constructs a RegressionBaseSelector.

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
        super().__init__(nickname)
        if self.nickname is None:
            self.nickname = f'KBestSelector({scorer})'
        self.scorer = scorer


    def select(self, df: pd.DataFrame, X_vars: list[str], y_var: str, 
               n_target_features: int):
        """
        Selects the top n_target_features features.

        Parameters
        ----------
        - df : pd.DataFrame.
        - X_vars : list[str].
            A list of features to look through. 
        - y_var : str.
            The variable to be predicted.
        - n_target_features : int. 
            Number of desired features, < len(X_vars).

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
        selector.fit(
            X=df[X_vars],
            y=df[y_var]
        )
        self.selected_features = selector.get_feature_names_out()
        self.all_feature_scores = selector.scores_
        self.support = selector.get_support()
        self.selected_feature_scores = selector.scores_[self.support]
        return self.selected_features, self.support




class SimpleLinearSelectorR(RegressionBaseSelector):
    """Selects the (at most) k best features via Lasso regression model-inherent 
    feature selection.
    """

    def __init__(self, regularization_type: Literal[None, 'l1', 'l2'] = None, 
                 alpha = 0.0, nickname: str = None):
        """
        Constructs an SimpleLinearSelector.

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
        super().__init__(nickname)
        if self.nickname is None:
            self.nickname = f'LinearSelector({regularization_type}, {alpha})'
        if regularization_type == 'l1':
            self.model = Lasso(alpha=alpha)
        elif regularization_type == 'l2':
            self.model = Ridge(alpha=alpha)
        elif regularization_type == None:
            self.model = LinearRegression()
        else:
            raise ValueError(f'Invalid input: regularization_type = ' + \
                             f'{regularization_type}')

    def select(self, df: pd.DataFrame, X_vars: list[str], y_var: str, 
               n_target_features: int):
        """
        Selects (at maximum) the top n_target_features features.

        Parameters
        ----------
        - df : pd.DataFrame.
        - X_vars : list[str].
            A list of features to look through. 
        - y_var : str.
            The variable to be predicted.
        - n_target_features : int. 
            Number of desired features, < len(X_vars).

        Returns
        -------
        - np array ~ (n_features).
            Selected features.
        - np array ~ (n_features).
            Boolean mask.
        """
        self.model.fit(
            X=df[X_vars].to_numpy(),
            y=df[y_var].to_numpy()
        )
        selector = SelectFromModel(estimator=self.model, prefit=True, 
            max_features=n_target_features)
        selector.fit(
            X=df[X_vars],
            y=df[y_var]
        )
        self.selected_features = selector.get_feature_names_out()
        self.support = selector.get_support()
        return self.selected_features, self.support
    


class RFESelectorR(RegressionBaseSelector):
    """Selects the k best features via L1-based 
    recursive feature elimination.
    """

    def __init__(self, model: Literal['ols', 'l1', 'l2', 'decision_tree', 
                    'svm', 'random_forest'] | BaseEstimator, 
                 nickname: str = None):
        """
        Constructs an RFESelector. 

        Parameters
        ----------
        - model : str | BaseEstimator.
        - nickname : str.
            Default: None. If None, then outputs the class name. 

        Returns
        -------
        - None
        """
        super().__init__(nickname)
        if self.nickname is None:
            self.nickname = f"RFESelector({model})"
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


    def select(self, df: pd.DataFrame, X_vars: list[str], y_var: str, 
               n_target_features: int):
        """
        Selects (at maximum) the top n_target_features features.

        Parameters
        ----------
        - df : pd.DataFrame.
        - X_vars : list[str].
            A list of features to look through. 
        - y_var : str.
            The variable to be predicted.
        - n_target_features : int. 
            Number of desired features, < len(X_vars).

        Returns
        -------
        - np array ~ (n_features).
            Selected features.
        - np array ~ (n_features).
            Boolean mask.
        """
        selector = RFE(estimator=self.model, 
                       n_features_to_select=n_target_features)
        selector.fit(
            X=df[X_vars],
            y=df[y_var]
        )
        self.selected_features = selector.get_feature_names_out()
        self.support = selector.get_support()
        return self.selected_features, self.support


