import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    BaggingRegressor
)
from .base import BaseRegression


class TreeRegression(BaseRegression):
    """A simple decision tree regressor. 
    
    Like all BaseRegression-derived classes, hyperparameter selection is 
    performed automatically during training. The cross validation and 
    hyperparameter selection process can be modified by the user. 
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, random_state: int):
        """
        Initializes a TreeRegression object. 

        Parameters
        ----------
        - X : np.ndarray ~ (n_samples, n_regressors).
            Matrix of predictor variables. 
        - y : np.ndarray ~ (n_samples).
        - random_state: int. 

        Returns
        -------
        - None
        """
        super().__init__(X, y)
        self.random_state = random_state
    
    def fit(self, X: np.ndarray = None, y: np.ndarray = None, 
            random_state: int = None):
        """Fits the model. 

        Parameters
        ----------
        - X : np.ndarray ~ (n_samples, n_regressors).
            Default: None.
        - y : np.ndarray ~ (n_samples).
            Default: None.
        - random_state : int. 
            Default: None. 

        Returns
        -------
        - None
        """
        super().fit(X, y)
        if random_state is not None:
            self.random_state = random_state
        self._hyperparam_selector = GridSearchCV


class TreeEnsembleRegression(BaseRegression):
    """Ensemble tree regressor. Includes random forest, gradient boosting, 
    and bagging. 
    
    Like all BaseRegression-derived classes, hyperparameter selection is 
    performed automatically during training. The cross validation and 
    hyperparameter selection process can be modified by the user. 
    """


