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

    def __init__(self, type: Literal['ols', 'l1', 'l2'] = 'ols', 
                 hyperparam_search_method: \
                    Literal[None, 'grid', 'random'] = None, 
                 hyperparam_grid_specification: Mapping[str, Iterable] = None,
                 model_random_state: int = 42,
                 id: str = None, **kwargs):
        """
        Initializes a LinearR object. 

        Parameters
        ----------
        - type : Literal['ols', 'l1', 'l2']. 
            Default: 'ols'.
        - hyperparam_search_method : str. 
            Default: None. If None, a regression-specific default hyperparameter 
            search is conducted. 
        - hyperparam_grid_specification : Mapping[str, list]. 
            Default: None. If None, a regression-specific default hyperparameter 
            search is conducted. 
        - id : str. 
            Default: None. Determines how the model shows up in the reports. 
            If None, the id is set to be the class name.
        - model_random_state : int.
            Default: 42. Random seed for the model.
        - kwargs : Key word arguments are passed directly into the 
            intialization of the HyperparameterSearcher class. In particular, 
            inner_cv and inner_random_state can be set via kwargs. 

        Notable kwargs
        --------------
        - inner_cv : int | BaseCrossValidator.
        - inner_cv_seed : int.
        """
        super().__init__()

        self._type = type
        if id is None:
            self._id = f'LinearR({self._type})'
        else:
            self._id = id

        if type == 'ols':
            self._estimator = LinearRegression()
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'fit_intercept': [True]
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )
        elif type == 'l1':
            self._estimator = Lasso(selection='random',
                random_state=model_random_state, max_iter=2000)
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'alpha': np.logspace(-5, 2, 100)
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )
        elif type == 'l2':
            self._estimator = Ridge(
                random_state=model_random_state, max_iter=2000)
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'alpha': np.logspace(-5, 2, 100)
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )
        else:
            raise ValueError('Invalid value for regularization_type')




class RobustLinearR(BaseRegression):
    """Robust linear regression.

    Like all classes extending BaseRegression, hyperparameter selection is 
    performed automatically during training. The cross validation and 
    hyperparameter selection process can be modified by the user. 
    """

    def __init__(self, type: Literal['huber', 'ransac'] = 'huber', 
                 hyperparam_search_method: \
                    Literal[None, 'grid', 'random'] = None, 
                 hyperparam_grid_specification: Mapping[str, Iterable] = None,
                 id: str = None, **kwargs):
        """
        Initializes a RobustLinearR object. 

        Parameters
        ----------
        - type : Literal['huber', 'ransac']. 
            Default: 'huber'.
        - hyperparam_search_method : Literal[None, 'grid', 'random']. 
            Default: None. If None, a regression-specific default hyperparameter 
            search is conducted. 
        - hyperparam_grid_specification : Mapping[str, list]. 
            Default: None. If None, a regression-specific default hyperparameter 
            search is conducted. 
        - id : str. 
            Default: None. Determines how the model shows up in the reports. 
            If None, the id is set to be the class name.
        - kwargs : Key word arguments are passed directly into the 
            intialization of the HyperparameterSearcher class. In particular, 
            inner_cv and inner_random_state can be set via kwargs. 

        Notable kwargs
        --------------
        - inner_cv : int | BaseCrossValidator. 
        - inner_cv_seed : int.
        """
        super().__init__()

        self._type = type

        if id is None:
            self._id = f'RobustLinearR({type})'
        else:
            self._id = id

        if type == 'huber':
            self._estimator = HuberRegressor(max_iter=1000)
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'epsilon': [1.0, 1.2, 1.35, 1.5, 2.0],
                    'alpha': np.logspace(-6, -1, num=10)
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )
        elif type == 'ransac':
            self._estimator = RANSACRegressor()
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'misample_size': [None, 0.1, 0.25, 0.5], 
                    'residual_threshold': [None, 1.0, 2.0], 
                    'max_trials': [100, 200, 300]
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )

