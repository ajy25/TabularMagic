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

    def __init__(self,
                 type: Literal['linear', 'poly', 'rbf'] = 'rbf', 
                 hyperparam_search_method: str = None, 
                 hyperparam_grid_specification: Mapping[str, Iterable] = None,
                 name: str = None, **kwargs):
        """
        Initializes a SVMR object. 

        Parameters
        ----------
        - type : Literal['linear', 'poly', 'rbf']. 
            Default: 'rbf'. The type of kernel to use.
        - hyperparam_search_method : str. 
            Default: None. If None, a regression-specific default hyperparameter 
            search is conducted. 
        - hyperparam_grid_specification : Mapping[str, list]. 
            Default: None. If None, a regression-specific default hyperparameter 
            search is conducted. 
        - name : str. 
            Default: None. Determines how the model shows up in the reports. 
            If None, the name is set to be the class name.
        - kwargs : Key word arguments are passed directly into the 
            intialization of the HyperparameterSearcher class. In particular, 
            inner_cv and inner_cv_seed can be set via kwargs. 
            
        Notable kwargs
        --------------
        - inner_cv : int | BaseCrossValidator.
        - inner_cv_seed : int.
        - n_jobs : int. Number of parallel jobs to run.
        - verbose : int. sklearn verbosity level.
        """
        super().__init__()

        if name is None:
            self._name = f'SVMR({type})'
        else:
            self._name = name

        self._estimator = SVR(kernel=type)


        if (hyperparam_search_method is None) or \
            (hyperparam_grid_specification is None):
            hyperparam_search_method = 'grid'

            if type == 'linear':
                hyperparam_grid_specification = {
                    'C': np.logspace(-2, 2, 10),
                }
            elif type == 'poly':
                hyperparam_grid_specification = {
                    'C': np.logspace(-2, 2, 5),
                    'degree': [2, 3, 4, 5],
                    'gamma': np.logspace(-2, 2, 5),
                    'coef0': [0, 1, 2, 3],
                }
            elif type == 'rbf':
                hyperparam_grid_specification = {
                    'C': np.logspace(-4, 2, 10),
                    'gamma': np.logspace(-4, 2, 10),
                }

        self._hyperparam_searcher = HyperparameterSearcher(
            estimator=self._estimator,
            method=hyperparam_search_method,
            grid=hyperparam_grid_specification,
            **kwargs
        )


        

