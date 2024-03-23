import numpy as np 
from sklearn.neural_network import MLPRegressor
from typing import Mapping, Iterable
from .base_regression import BaseRegression, HyperparameterSearcher


class MLP(BaseRegression):
    """Support Vector Machine with kernel trick.
    
    Like all BaseRegression-derived classes, hyperparameter selection is 
    performed automatically during training. The cross validation and 
    hyperparameter selection process can be modified by the user. 
    """

    def __init__(self, X: np.ndarray = None, y: np.ndarray = None, 
                 hyperparam_search_method: str = None, 
                 hyperparam_grid_specification: Mapping[str, Iterable] = None,
                 nickname: str = None, **kwargs):
        """
        Initializes an MLP object. 

        Parameters
        ----------
        - X : np.ndarray ~ (n_samples, n_regressors).
            Default: None. Matrix of predictor variables. 
        - y : np.ndarray ~ (n_samples).
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
            intialization of the hyperparameter search method. 

        Returns
        -------
        - None
        """
        super().__init__(X, y)

        if nickname is None:
            self.nickname = 'MLP'
        else:
            self.nickname = nickname

        self.estimator = MLPRegressor()
        if (hyperparam_search_method is None) or \
            (hyperparam_grid_specification is None):
            hyperparam_search_method = 'grid'
            hyperparam_grid_specification = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50, 25)],
                'activation': ['relu'],
                'solver': ['adam'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
            }
        self._hyperparam_searcher = HyperparameterSearcher(
            estimator=self.estimator,
            method=hyperparam_search_method,
            grid=hyperparam_grid_specification,
            **kwargs
        )


        

