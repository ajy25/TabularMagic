import numpy as np
from typing import Literal, Mapping, Iterable
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    GridSearchCV, 
    RandomizedSearchCV
)
from skopt import BayesSearchCV
from ...metrics import RegressionScorer
from ..base_model import BaseModel, HyperparameterSearcher



class BaseRegression(BaseModel):
    """A class that provides the framework upon which all regression 
    objects are built. 

    BaseRegression wraps sklearn methods. 
    The primary purpose of BaseRegression is to automate the scoring and 
    model selection processes. 
    """

    def __init__(self, X: np.ndarray = None, y: np.ndarray = None):
        """Initializes a BaseRegression object. Creates copies of the inputs. 

        Parameters
        ----------
        - X : np.ndarray ~ (n_samples, n_regressors).
            Default: None. Matrix of predictor variables. 
        - y : np.ndarray ~ (n_samples).
            Default: None. Dependent variable vector. 
        Returns
        -------
        - None
        """
        if (X is not None) and (y is not None):
            self._X = X.copy()
            self._y = y.copy()
            self._n_samples = X.shape[0]
            self._n_regressors = X.shape[1]
        self._hyperparam_searcher: HyperparameterSearcher = None
        self.estimator: BaseEstimator = None

    def fit(self, X: np.ndarray = None, y: np.ndarray = None):
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
        if (X is not None) and (y is not None):
            self._X = X.copy()
            self._y = y.copy()
            self._n_samples = X.shape[0]
            self._n_regressors = X.shape[1]
        if (self._X is None) or (self._y is None):
            raise ValueError(f'Invalid input: X, y.',
                             'X and y both must not be None.')
        self._verify_Xy_input_validity(self._X, self._y)
        self._hyperparam_searcher.fit(self._X, self._y)
        self.estimator = self._hyperparam_searcher.best_estimator
        
    def predict(self, X: np.ndarray):
        """Returns y_pred.
        
        Parameters
        ----------
        - X : np.ndarray ~ (n_test_samples, n_regressors).

        Returns
        -------
        - np.ndarray ~ (n_test_samples)
        """
        self._verify_X_input_validity(X)
        return self.estimator.predict(X)

    def score(self, X_test: np.ndarray = None, y_test: np.ndarray = None) -> \
            RegressionScorer:
        """Returns the MSE, MAD, Pearson Correlation, and Spearman Correlation 
        of the true and predicted y values. Also, returns the R-squared and 
        adjusted R-squared 

        Parameters
        ----------
        - X : np.ndarray ~ (n_samples, n_regressors).
            Default: None. If None, computes scores using X and y. 
        - y : np.ndarray ~ (n_samples).
            Default: None. If None, computes scores using X and y. 

        Returns
        -------
        - RegressionScorer. 
        """
        if X_test is None or y_test is None:
            X_test = self._X
            y_test = self._y
        self._verify_Xy_input_validity(X_test, y_test)
        return RegressionScorer(self.predict(X_test), y_test, 
            n_regressors=self._n_regressors, model_id_str=str(self))
    
    def _verify_Xy_input_validity(self, X: np.ndarray, y: np.ndarray):
        """Verifies that the inputs X and y are valid. If invalid, raises 
        the appropriate error with the appropriate error message. 

        Parameters
        ----------
        - X : np.ndarray ~ (n_samples, n_regressors).
        - y : np.ndarray ~ (n_samples).

        Returns
        -------
        - None
        """
        if not isinstance(X, np.ndarray):
            raise ValueError(f'Invalid input: X. Must be 2d np array.')
        if not isinstance(y, np.ndarray):
            raise ValueError(f'Invalid input: y. Must be 1d np array.')
        if len(X.shape) != 2:
            raise ValueError(f'Invalid input: X. Must be 2d np array.')
        if len(y.shape) != 1:
            raise ValueError(f'Invalid input: y. Must be 1d np array.')
        if X.shape[0] != y.shape[0]:
            raise ValueError(f'Invalid input: X, y. Must have the same',
                             'length in the first dimension.')

    def _verify_X_input_validity(self, X: np.ndarray):
        """Verifies that the input X is. If invalid, raises 
        the appropriate error with the appropriate error message. 

        Parameters
        ----------
        - X : np.ndarray ~ (n_samples, n_regressors).

        Returns
        -------
        - None
        """
        if not isinstance(X, np.ndarray):
            raise ValueError(f'Invalid input: X. Must be 2d np array.')
        if X.shape[1] != self._n_regressors:
            raise ValueError(f'Invalid input: X. Must have the same',
                             'length in the second dimension as the dataset',
                             'upon which the estimator has been trained.')

    def __str__(self):
        return 'BaseRegression'

