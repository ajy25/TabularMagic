import numpy as np
from sklearn.model_selection import (
    GridSearchCV, 
    RandomizedSearchCV,
    KFold,
    BaseCrossValidator
)
from typing import Literal, Mapping, Iterable
from sklearn.base import BaseEstimator
from sklearn.utils._testing import ignore_warnings


class BaseDiscriminativeModel():
    """Skeletal class for typing assistance of BaseRegression and 
    BaseClassification. 
    
    BaseRegression and BaseClassification extend BaseDiscriminativeModel. 
    BaseDiscriminativeModel has no funtionality beyond providing 
    typing assistance elsewhere. 
    """

    def __init__(self):
        self.nickname = 'BaseDiscriminativeModel'

    def fit(self, X, y):
        """Fits the model. 

        Parameters
        ----------
        - X : np.ndarray ~ (sample_size, n_predictors).
            Default: None.
        - y : np.ndarray ~ (sample_size).
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
        - X : np.ndarray ~ (sample_size, n_predictors).
            Default: None. If None, computes scores using X and y. 
        - y : np.ndarray ~ (sample_size).
            Default: None. If None, computes scores using X and y. 

        Returns
        -------
        - Scoring object. 
        """
        pass

    def __str__(self):
        return self.nickname
    
    def save_checkpoint(self, dirpath: str = None, filepath: str = None):
        pass





class HyperparameterSearcher():
    """A wrapper for common hyperparameter search methods.
    """

    def __init__(self, estimator: BaseEstimator, 
                 method: Literal['grid', 'random'], 
                 grid: Mapping[str, Iterable], 
                 inner_cv: int | BaseCrossValidator = 5,
                 inner_cv_seed: int = 42,
                 **kwargs):
        """Initializes a HyperparameterSearch object.
        
        Parameters
        ----------
        - estimator : sklearn.base.BaseEstimator
        - method : str. 
            Must be an element in ['grid', 'random', 'bayes']. 
        - grid : dict.
            Specification of the set/distribution of hypeparameters to 
            search through. 
        - cv : int | BaseCrossValidator.
            Default: 5-fold cross validation.
        - inner_cv_seed : int.
        - kwargs. 
            Key word arguments are passed directly into the intialization of the 
            hyperparameter search method. 
        
        Returns
        -------
        - None
        """
        
        self.best_estimator = None
        if isinstance(inner_cv, int):
            self.inner_cv = KFold(n_splits=inner_cv, 
                                  random_state=inner_cv_seed, 
                                  shuffle=True)
        elif isinstance(inner_cv, BaseCrossValidator):
            self.inner_cv = inner_cv
        if method == 'grid':
            self._searcher = GridSearchCV(estimator, grid, cv=self.inner_cv, 
                                          **kwargs)
        elif method == 'random':
            self._searcher = RandomizedSearchCV(estimator, grid, 
                                                cv=self.inner_cv, **kwargs)
        else:
            raise ValueError('Invalid input: method.')


    def fit(self, X: np.ndarray, y: np.ndarray):
        """Cross validation search of optimal hyperparameters. Idenfities 
        best estimator. 

        Parameters
        ----------
        - X : np.ndarray ~ (sample_size, n_predictors).
        - y : np.ndarray ~ (sample_size).

        Returns
        -------
        - best_estimator : BaseEstimator. 
        """
        ignore_warnings(self._searcher.fit)(X, y)
        self.best_estimator = self._searcher.best_estimator_
        self.best_params = self._searcher.best_params_
        return self.best_estimator



