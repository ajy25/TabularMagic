import numpy as np 
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, 
    HuberRegressor, RANSACRegressor)
from typing import Mapping, Literal, Iterable
from .base_regression import BaseRegression, HyperparameterSearcher



class LinearR(BaseRegression):
    """(Regularized) Least Squares regressor. 
    
    Like all BaseRegression-derived classes, hyperparameter selection is 
    performed automatically during training. The cross validation and 
    hyperparameter selection process can be modified by the user. 
    """

    def __init__(self, X: np.ndarray = None, y: np.ndarray = None, 
                 regularization_type: Literal['OLS', 'l1', 'l2'] = 'OLS', 
                 hyperparam_search_method: \
                    Literal[None, 'grid', 'random'] = None, 
                 hyperparam_grid_specification: Mapping[str, Iterable] = None,
                 nickname: str = None, **kwargs):
        """
        Initializes a Linear object. 

        Parameters
        ----------
        - X : np.ndarray ~ (n_samples, n_regressors).
            Default: None. Matrix of predictor variables. 
        - y : np.ndarray ~ (n_samples).
            Default: None. Dependent variable vector. 
        - regularization_type : [OLS, 'l1', 'l2']. 
            Default: None.
        - hyperparam_search_method : str. 
            Default: None. If None, a regression-specific default hyperparameter 
            search is conducted. 
        - hyperparam_grid_specification : Mapping[str, list]. 
            Default: None. If None, a regression-specific default hyperparameter 
            search is conducted. 
        - nickname : str. 
            Default: None. Determines how the model shows up in the reports. 
            If None, the nickname is set to be the class name.
        - kwargs : Key word arguments are passed directly into the 
            intialization of the HyperparameterSearcher class. In particular, 
            inner_cv and inner_random_state can be set via kwargs. 

        Returns
        -------
        - None
        """
        super().__init__(X, y)
        self.regularization_type = regularization_type

        if nickname is None:
            self.nickname = f'LinearR({regularization_type})'
        else:
            self.nickname = nickname

        if regularization_type == 'OLS':
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
                grid=hyperparam_grid_specification,
                **kwargs
            )
        elif regularization_type == 'l1':
            self.estimator = Lasso(random_state=42, max_iter=2000)
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'alpha': np.logspace(-5, 1, 100)
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self.estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )
        elif regularization_type == 'l2':
            self.estimator = Ridge(random_state=42, max_iter=2000)
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'alpha': np.logspace(-5, 1, 100)
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self.estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )
        else:
            raise ValueError('Invalid value for regularization_type')




class RobustLinearR(BaseRegression):
    """Robust linear regression.
    """
    def __init__(self, X: np.ndarray = None, y: np.ndarray = None, 
                 regressor_type: Literal['huber', 'ransac'] = 'huber', 
                 hyperparam_search_method: \
                    Literal[None, 'grid', 'random'] = None, 
                 hyperparam_grid_specification: Mapping[str, Iterable] = None,
                 nickname: str = None, **kwargs):
        """
        Initializes a Linear object. 

        Parameters
        ----------
        - X : np.ndarray ~ (n_samples, n_regressors).
            Default: None. Matrix of predictor variables. 
        - y : np.ndarray ~ (n_samples).
            Default: None. Dependent variable vector. 
        - regressor : Literal['huber', 'ransac']. 
            Default: 'huber'.
        - hyperparam_search_method : Literal[None, 'grid', 'random']. 
            Default: None. If None, a regression-specific default hyperparameter 
            search is conducted. 
        - hyperparam_grid_specification : Mapping[str, list]. 
            Default: None. If None, a regression-specific default hyperparameter 
            search is conducted. 
        - nickname : str. 
            Default: None. Determines how the model shows up in the reports. 
            If None, the nickname is set to be the class name.
        - kwargs : Key word arguments are passed directly into the 
            intialization of the HyperparameterSearcher class. In particular, 
            inner_cv and inner_random_state can be set via kwargs. 

        Returns
        -------
        - None
        """
        super().__init__(X, y)
        self.regressor_type = regressor_type

        if nickname is None:
            self.nickname = f'RobustLinearR({regressor_type})'
        else:
            self.nickname = nickname

        if regressor_type == 'huber':
            self.estimator = HuberRegressor(max_iter=1000)
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'epsilon': [1.0, 1.2, 1.35, 1.5, 2.0],
                    'alpha': np.logspace(-6, -1, num=10)
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self.estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )
        elif regressor_type == 'ransac':
            self.estimator = RANSACRegressor()
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'min_samples': [None, 0.1, 0.25, 0.5], 
                    'residual_threshold': [None, 1.0, 2.0], 
                    'max_trials': [100, 200, 300]
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self.estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )

