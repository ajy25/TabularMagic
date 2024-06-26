import numpy as np 
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, 
    ElasticNet, HuberRegressor, RANSACRegressor)
from typing import Mapping, Literal, Iterable
from .base import BaseR, HyperparameterSearcher



class LinearR(BaseR):
    """Class for linear regression (optionally with penalty).
    
    Like all BaseR-derived classes, hyperparameter selection is 
    performed automatically during training. The cross validation and 
    hyperparameter selection process can be modified by the user. 
    """

    def __init__(self, type: Literal['ols', 'l1', 'l2', 'elasticnet'] = 'ols', 
                 hyperparam_search_method: \
                    Literal[None, 'grid', 'random'] = None, 
                 hyperparam_grid_specification: Mapping[str, Iterable] = None,
                 model_random_state: int = 42,
                 name: str = None, **kwargs):
        """
        Initializes a LinearR object. 

        Parameters
        ----------
        type : Literal['ols', 'l1', 'l2', 'elasticnet']. 
            Default: 'ols'. The type of linear regression to be used.
        hyperparam_search_method : Literal[None, 'grid', 'random']. 
            Default: None. If None, a regression-specific default hyperparameter 
            search is conducted. 
        hyperparam_grid_specification : Mapping[str, list]. 
            Default: None. If None, a regression-specific default hyperparameter 
            search is conducted. 
        model_random_state : int.
            Default: 42. Random seed for the model.
        name : str. 
            Default: None. Determines how the model shows up in the reports. 
            If None, the name is set to be the class name.
        kwargs : Key word arguments are passed directly into the 
            intialization of the HyperparameterSearcher class. In particular, 
            inner_cv and inner_cv_seed can be set via kwargs. 

        **kwargs
        --------------
        inner_cv : int | BaseCrossValidator.
            Default: 5.
        inner_cv_seed : int.
            Default: 42.
        n_jobs : int. 
            Default: 1. Number of parallel jobs to run.
        verbose : int. 
            Default: 0. scikit-learn verbosity level.
        """
        super().__init__()
        self._dropfirst = True

        self._type = type
        if name is None:
            self._name = f'LinearR({self._type})'
        else:
            self._name = name

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
                random_state=model_random_state)
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
                random_state=model_random_state)
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
        elif type == 'elasticnet':
            self._estimator = ElasticNet(selection='random',
                random_state=model_random_state)
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'alpha': np.logspace(-5, 2, 10),
                    'l1_ratio': np.linspace(0, 1, 10)
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )
        else:
            raise ValueError(f'Invalid value for type: {type}.')




class RobustLinearR(BaseR):
    """Class for robust linear regression.

    Like all classes extending BaseRegression, hyperparameter selection is 
    performed automatically during training. The cross validation and 
    hyperparameter selection process can be modified by the user. 
    """

    def __init__(self, type: Literal['huber', 'ransac'] = 'huber', 
                 hyperparam_search_method: \
                    Literal[None, 'grid', 'random'] = None, 
                 hyperparam_grid_specification: Mapping[str, Iterable] = None,
                 model_random_state: int = 42,
                 name: str = None, **kwargs):
        """
        Initializes a RobustLinearR object. 

        Parameters
        ----------
        type : Literal['huber', 'ransac'].
            Default: 'huber'.
        hyperparam_search_method : Literal[None, 'grid', 'random']. 
            Default: None. If None, a regression-specific default hyperparameter 
            search is conducted. 
        hyperparam_grid_specification : Mapping[str, list]. 
            Default: None. If None, a regression-specific default hyperparameter 
            search is conducted. 
        model_random_state : int.
            Default: 42. Random seed for the model.
        name : str. 
            Default: None. Determines how the model shows up in the reports. 
            If None, the name is set to be the class name.
        kwargs : Key word arguments are passed directly into the 
            intialization of the HyperparameterSearcher class. In particular, 
            inner_cv and inner_cv_seed can be set via kwargs. 

        **kwargs
        --------------
        inner_cv : int | BaseCrossValidator.
            Default: 5.
        inner_cv_seed : int.
            Default: 42.
        n_jobs : int. 
            Default: 1. Number of parallel jobs to run.
        verbose : int. 
            Default: 0. scikit-learn verbosity level.
        """

        super().__init__()
        self._dropfirst = True

        self._type = type

        if name is None:
            self._name = f'RobustLinearR({type})'
        else:
            self._name = name

        if type == 'huber':
            self._estimator = HuberRegressor()
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'epsilon': [1.0, 1.2, 1.35, 1.5, 2.0],
                    'alpha': np.logspace(-5, 2, num=10)
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )
        elif type == 'ransac':
            self._estimator = RANSACRegressor(random_state=model_random_state)
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'min_samples': [None, 0.1, 0.25, 0.5], 
                    'residual_threshold': [None, 1.0, 2.0], 
                    'max_trials': [100, 200, 300]
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )

        else:
            raise ValueError(f'Invalid value for type: {type}.')
        

