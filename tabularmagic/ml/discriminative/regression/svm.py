import numpy as np 
from sklearn.svm import SVR
from typing import Mapping, Iterable, Literal
from .base_regression import BaseRegression, HyperparameterSearcher

class SVMR(BaseRegression):
    """Support Vector Machine with kernel trick.
    
    Like all BaseRegression-derived classes, hyperparameter selection is 
    performed automatically during training. The cross validation and 
    hyperparameter selection process can be modified by the user. 
    """

    def __init__(self, X: np.ndarray = None, y: np.ndarray = None, 
                 kernel: Literal['linear', 'poly', 'rbf'] = 'rbf', 
                 hyperparam_search_method: str = None, 
                 hyperparam_grid_specification: Mapping[str, Iterable] = None,
                 nickname: str = None, **kwargs):
        """
        Initializes a SVMR object. 

        Parameters
        ----------
        - X : np.ndarray ~ (sample_size, n_predictors).
            Default: None. Matrix of predictor variables. 
        - y : np.ndarray ~ (sample_size).
            Default: None. Dependent variable vector. 
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

        if nickname is None:
            self.nickname = f'SVMR({kernel})'
        else:
            self.nickname = nickname

        self.estimator = SVR(kernel=kernel)


        if (hyperparam_search_method is None) or \
            (hyperparam_grid_specification is None):
            hyperparam_search_method = 'grid'

            if kernel == 'linear':
                # TODO: fill out a good guess
                pass
            elif kernel == 'poly':
                # TODO: fill out a good guess
                pass
            elif kernel == 'rbf':
                hyperparam_grid_specification = {
                    'C': np.logspace(-4, 2, 10),
                    'gamma': np.logspace(-4, 2, 10),
                }

        self._hyperparam_searcher = HyperparameterSearcher(
            estimator=self.estimator,
            method=hyperparam_search_method,
            grid=hyperparam_grid_specification,
            **kwargs
        )


        

