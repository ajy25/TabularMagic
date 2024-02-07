from sklearn.feature_selection import (SelectKBest, f_regression, 
    mutual_info_regression, r_regression, RFE, SelectFromModel)
from sklearn.linear_model import Lasso
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





class KBestSelector(RegressionBaseSelector):
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
            self.nickname = f"KBestSelector({scorer})"
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




class L1RegSelector(RegressionBaseSelector):
    """Selects the (at most) k best features based on Lasso regression
    """

    def __init__(self, alpha = 0.1, nickname: str = None):
        """
        Constructs an L1RegSelector.

        Parameters
        ----------
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
            self.nickname = f"L1RegSelector({alpha})"
        self.model = Lasso(alpha=alpha)
        

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
    



