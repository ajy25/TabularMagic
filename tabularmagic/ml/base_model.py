import numpy as np
from sklearn.model_selection import (
    GridSearchCV, 
    RandomizedSearchCV
)
from skopt import BayesSearchCV
from typing import Literal, Mapping, Iterable
from sklearn.base import BaseEstimator
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


class BaseModel():
    """Skeletal class for typing assistance of BaseRegression and 
    BaseClassification. 
    
    BaseRegression and BaseClassification extend BaseModel. BaseModel 
    has no funtionality beyond providing typing assistance elsewhere. 
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        """Fits the model. 

        Parameters
        ----------
        - X : np.ndarray ~ (n_samples, n_regressors).
            Default: None.
        - y : np.ndarray ~ (n_samples).
            Default: None.

        Returns
        -------
        - None
        """
        pass
    
    def score(self, X_test, y_test):
        """
        Parameters
        ----------
        - X : np.ndarray ~ (n_samples, n_regressors).
            Default: None. If None, computes scores using X and y. 
        - y : np.ndarray ~ (n_samples).
            Default: None. If None, computes scores using X and y. 

        Returns
        -------
        - Scoring object. 
        """
        pass

    def __str__(self):
        return 'BaseModel'


class HyperparameterSearcher():
    """A wrapper for common hyperparameter search methods.
    """

    def __init__(self, estimator: BaseEstimator, 
                 method: Literal['grid', 'random'], 
                 grid: Mapping[str, Iterable], **kwargs):
        """Initializes a HyperparameterSearch object.
        
        Parameters
        ----------
        - estimator : sklearn.base.BaseEstimator
        - method : str. 
            Must be an element in ['grid', 'random']. 
        - grid : dict.
            Specification of the set/distribution of hypeparameters to 
            search through. 
        - kwargs. 
            Key word arguments are passed directly into the intialization of the 
            hyperparameter search method. 
        
        Returns
        -------
        - None
        """
        self.best_estimator = None
        if method == 'grid':
            self._searcher = GridSearchCV(estimator, grid, **kwargs)
        elif method == 'random':
            self._searcher = RandomizedSearchCV(estimator, grid, **kwargs)
        elif method == 'bayes':
            self._searcher = BayesSearchCV(estimator, grid, **kwargs)


    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Cross validation search of optimal hyperparameters. Idenfities 
        best estimator. 

        Parameters
        ----------
        - X : np.ndarray ~ (n_samples, n_regressors).
        - y : np.ndarray ~ (n_samples).

        Returns
        -------
        - best_estimator : BaseEstimator. 
        """
        self._searcher.fit(X, y)
        self.best_estimator = self._searcher.best_estimator_
        return self.best_estimator



