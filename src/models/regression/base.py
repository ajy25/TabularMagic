import numpy as np
from ...metrics import RegressionScorer
from sklearn.model_selection import (
    GridSearchCV, 
    RandomizedSearchCV
)


class BaseRegression():
    """BaseRegression : Class that provides the framework upon which all 
    regression objects are built. 
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initializes a BaseRegression object. Creates copies of the inputs. 

        Parameters
        ----------
        - X : np.ndarray ~ (n_samples, n_regressors).
            Matrix of predictor variables. 
        - y : np.ndarray ~ (n_samples).

        Returns
        -------
        - None
        """
        self._X = X.copy()
        self._y = y.copy()
        self._n_samples = X.shape[0]
        self._n_regressors = X.shape[1]
        self._hyperparam_selector = None
        self._best_estimator = None

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
        self._verify_Xy_input_validity(X, y)
        if (X is not None) and (y is not None):
            self._X = X.copy()
            self._y = y.copy()
            self._n_samples = X.shape[0]
            self._n_regressors = X.shape[1]
    
    def predict(self, X: np.ndarray):
        """Returns y_pred.
        
        Parameters
        ----------
        - X : np.ndarray ~ (n_test_samples, n_regressors).

        Returns
        -------
        - None
        """
        self._verify_X_input_validity(X)

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
        self._verify_Xy_input_validity(X_test, y_test)
        return RegressionScorer(self.predict(X_test), y_test)
    
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



class HyperparameterSearchWrapper():
    """A wrapper for common hyperparameter search methods.
    """


