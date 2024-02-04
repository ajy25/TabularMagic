import numpy as np 
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from typing import Mapping, Literal, Iterable
from .base_regression import BaseRegression, HyperparameterSearcher



class Linear(BaseRegression):
    """(Regularized) Least Squares regressor. 
    
    Like all BaseRegression-derived classes, hyperparameter selection is 
    performed automatically during training. The cross validation and 
    hyperparameter selection process can be modified by the user. 
    """

    def __init__(self, X: np.ndarray = None, y: np.ndarray = None, 
                 regularization_type: Literal[None, 'l1', 'l2'] = None, 
                 hyperparam_search_method: str = None, 
                 hyperparam_grid_specification: Mapping[str, Iterable] = None,
                 **kwargs):
        """
        Initializes a TreeRegression object. 

        Parameters
        ----------
        - X : np.ndarray ~ (n_samples, n_regressors).
            Default: None. Matrix of predictor variables. 
        - y : np.ndarray ~ (n_samples).
            Default: None. Dependent variable vector. 
        - regularization_type : [None, 'l1', 'l2']. 
            Default: None.
        - hyperparam_search_method : str. 
            Default: None. If None, a regression-specific default hyperparameter 
            search is conducted. 
        - hyperparam_grid_specification : Mapping[str, list]. 
            Default: None. If None, a regression-specific default hyperparameter 
            search is conducted. 
        - kwargs : Key word arguments are passed directly into the 
            intialization of the hyperparameter search method. 

        Returns
        -------
        - None
        """
        super().__init__(X, y)
        self.regularization_type = regularization_type
        if regularization_type is None:
            self.estimator = LinearRegression()
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'fit_intercept': [True]
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self.estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification
            )
        elif regularization_type == 'l1':
            self.estimator = Lasso()
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'bayes'
                hyperparam_grid_specification = {
                    'alpha': (1e-6, 1e+6, 'log-uniform')
                }
            if 'n_iter' not in kwargs:
                kwargs['n_iter'] = 32
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self.estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )
        elif regularization_type == 'l2':
            self.estimator = Ridge()
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'bayes'
                hyperparam_grid_specification = {
                    'alpha': (1e-6, 1e+6, 'log-uniform')
                }
            if 'n_iter' not in kwargs:
                kwargs['n_iter'] = 32
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self.estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )

    def __str__(self):
        if self.regularization_type is None:
            return f'Linear'
        elif self.regularization_type == 'l1':
            return f'Lasso'
        elif self.regularization_type == 'l2':
            return f'Ridge'


