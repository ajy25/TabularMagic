import numpy as np 
from sklearn.svm import SVR
from typing import Mapping, Iterable, Literal
from .base_regression import BaseRegression, HyperparameterSearcher

class SVM(BaseRegression):
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
        Initializes a SVM object. 

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
            self.nickname = f'SVM({kernel})'
        else:
            self.nickname = nickname

        self.estimator = SVR(kernel=kernel)
        if (hyperparam_search_method is None) or \
            (hyperparam_grid_specification is None):
            hyperparam_search_method = 'bayes'
            hyperparam_grid_specification = {
                'C': (1e-6, 1e6, 'log-uniform'),
                'gamma': (1e-6, 1e1, 'log-uniform'),
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
        return self.nickname
        

