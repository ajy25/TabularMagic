import numpy as np 
import pandas as pd
from typing import Literal
import statsmodels.api as sm
from ...metrics.regression_scoring import RegressionScorer


class OrdinaryLeastSquares:
    """Statsmodels OLS wrapper.
    """

    def __init__(self, X: pd.DataFrame = None, 
                 y: pd.Series = None, 
                 regularization_type: Literal[None, 'l1', 'l2'] = None,
                 alpha: float = 0,
                 nickname: str = None):
        """
        Initializes a OrdinaryLeastSquares object. Regresses y on X.

        Parameters
        ----------
        - X : pd.DataFrame ~ (n_samples, n_regressors).
            Default: None. DataFrame of predictor variables. 
        - y : pd.Series ~ (n_samples).
            Default: None. Dependent variable series. 
        - regularization_type : [None, 'l1', 'l2']. 
            Default: None.
        - alpha : float.
            Default: 0.
        - nickname : str. 
            Default: None. Determines how the model shows up in the reports. 
            If None, the nickname is set to be the class name.
        """
        self.estimator = None
        self.regularization_type = regularization_type
        self.alpha = alpha
        self.nickname = nickname
        if self.nickname is None:
            self.nickname = f'OrdinaryLeastSquares({regularization_type})'
        if (X is not None) and (y is not None):
            self._X = X.copy()
            self._y = y.copy()
            self._n_samples = X.shape[0]
            self._n_regressors = X.shape[1]
        
    def fit(self, X: pd.DataFrame = None, y: pd.Series = None):
        """
        Fits the model. 

        Parameters
        ----------
        - X : pd.DataFrame ~ (n_samples, n_regressors).
            Default: None. DataFrame of predictor variables. 
        - y : pd.Series ~ (n_samples).
            Default: None. Dependent variable series. 

        Returns
        -------
        - None
        """
        if (X is not None) and (y is not None):
            self._X = X.copy()
            self._y = y.copy()
            self._n_samples = X.shape[0]
            self._n_regressors = X.shape[1]
        if ((self._X is None) or (self._y is None)) and \
            (not ((self._X is None) and (self._y is None))):
            raise ValueError(f'Invalid input: X, y.',
                             'X and y both must not be None.')
        self._X = sm.add_constant(self._X)
        self._verify_Xy_input_validity(self._X, self._y)
        if self.regularization_type is None:
            self.estimator = sm.OLS(self._y, self._X).fit(cov_type='HC3')
        else:
            if self.regularization_type == 'l1':
                self.estimator = sm.OLS(self._y, self._X).\
                    fit_regularized(alpha=self.alpha, L1_wt=1)
            elif self.regularization_type == 'l2':
                self.estimator = sm.OLS(self._y, self._X).\
                    fit_regularized(alpha=self.alpha, L1_wt=0)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns y_pred.
        
        Parameters
        ----------
        - X : pd.DataFrame ~ (n_test_samples, n_regressors).

        Returns
        -------
        - np.ndarray ~ (n_test_samples)
        """
        X = sm.add_constant(X)
        self._verify_X_input_validity(X)
        return self.estimator.predict(X).to_numpy()
    
    def score(self, X_test: pd.DataFrame = None, y_test: pd.Series = None) \
        -> RegressionScorer:
        """Returns the MSE, MAD, Pearson Correlation, and Spearman Correlation 
        of the true and predicted y values. Also, returns the R-squared and 
        adjusted R-squared 

        Parameters
        ----------
        - X : pd.DataFrame ~ (n_samples, n_regressors).
            Default: None. If None, computes scores using X and y. 
        - y : pd.Series ~ (n_samples).
            Default: None. If None, computes scores using X and y. 

        Returns
        -------
        - RegressionScorer. 
        """
        if X_test is None or y_test is None:
            X_test = self._X
            y_test = self._y
        return RegressionScorer(self.predict(X_test), y_test.to_numpy(), 
            n_regressors=self._n_regressors, model_id_str=str(self))
    

    def _verify_Xy_input_validity(self, X: pd.DataFrame, y: pd.Series):
        """Verifies that the inputs X and y are valid. If invalid, raises 
        the appropriate error with the appropriate error message. 

        Parameters
        ----------
        - X : pd.DataFrame ~ (n_samples, n_regressors).
        - y : pd.Series ~ (n_samples).

        Returns
        -------
        - None
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f'Invalid input: X. Must be 2d pd.DataFrame.')
        if not isinstance(y, pd.Series):
            raise ValueError(f'Invalid input: y. Must be pd.Series.')
        if len(X.shape) != 2:
            raise ValueError(f'Invalid input: X. Must be 2d pd.DataFrame.')
        if len(y.shape) != 1:
            raise ValueError(f'Invalid input: y. Must be pd.Series.')
        if len(X) != len(y):
            raise ValueError(f'Invalid input: X, y. Must have the same ',
                             'length in the first dimension.')

    def _verify_X_input_validity(self, X: pd.DataFrame):
        """Verifies that the input X is valid. If invalid, raises 
        the appropriate error with the appropriate error message. 

        Parameters
        ----------
        - X : pd.DataFrame ~ (n_samples, n_regressors).

        Returns
        -------
        - None
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f'Invalid input: X. Must be pd.DataFrame.')
        # add 1 to n_regressors to account for constant
        if X.shape[1] != self._n_regressors + 1:
            raise ValueError(f'Invalid input: X. Must have the same ' + \
                             'length in the second dimension as the dataset' + \
                             ' upon which the estimator has been trained.')


    def __str__(self):
        return self.nickname


