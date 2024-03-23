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
                 hyperparam_search_method: \
                    Literal[None, 'grid', 'random', 'bayes'] = None, 
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
        - regularization_type : [None, 'l1', 'l2']. 
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
            self.nickname = f'Linear({regularization_type})'
        else:
            self.nickname = nickname

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



