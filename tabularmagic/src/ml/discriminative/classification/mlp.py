import numpy as np 
from sklearn.neural_network import MLPClassifier
from typing import Mapping, Iterable
from .base import BaseC, HyperparameterSearcher


class MLPC(BaseC):
    """Multi-layer Perceptron classifier.
    
    Like all BaseC-derived classes, hyperparameter selection is 
    performed automatically during training. The cross validation and 
    hyperparameter selection process can be modified by the user. 
    """

    def __init__(self, hyperparam_search_method: str = None, 
                 hyperparam_grid_specification: Mapping[str, Iterable] = None,
                 model_random_state: int = 42,
                 name: str = None, **kwargs):
        """
        Initializes an MLPC object. 

        Parameters
        ----------
        - hyperparam_search_method : str. 
            Default: None. If None, a 
            classification-specific default hyperparameter 
            search is conducted. 
        - hyperparam_grid_specification : Mapping[str, list]. 
            Default: None. If None, a classification-specific 
            default hyperparameter 
            search is conducted. 
        - name : str. 
            Default: None. Determines how the model shows up in the reports. 
            If None, the name is set to be the class name.
        - kwargs : Key word arguments are passed directly into the 
            intialization of the hyperparameter search method. 

        Notable kwargs
        --------------
        - inner_cv : int | BaseCrossValidator.
        - inner_cv_seed : int.
        - n_jobs : int. Number of parallel jobs to run.
        - verbose : int. sklearn verbosity level.
        """
        super().__init__()

        self._type = type
        if name is None:
            self._name = f'MLPC'
        else:
            self._name = name

        self._estimator = MLPClassifier(random_state=model_random_state)
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
            estimator=self._estimator,
            method=hyperparam_search_method,
            grid=hyperparam_grid_specification,
            **kwargs
        )


        

