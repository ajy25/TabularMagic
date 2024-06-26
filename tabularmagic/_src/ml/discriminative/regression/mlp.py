from sklearn.neural_network import MLPRegressor
from typing import Mapping, Iterable
from .base import BaseR, HyperparameterSearcher


class MLPR(BaseR):
    """Class for multi-layer perceptron regression.
    
    Like all BaseR-derived classes, hyperparameter selection is 
    performed automatically during training. The cross validation and 
    hyperparameter selection process can be modified by the user. 
    """

    def __init__(self, hyperparam_search_method: str = None, 
                 hyperparam_grid_specification: Mapping[str, Iterable] = None,
                 model_random_state: int = 42,
                 name: str = None, **kwargs):
        """
        Initializes an MLPR object. 

        Parameters
        ----------
        hyperparam_search_method : str. 
            Default: None. If None, a regression-specific default hyperparameter 
            search is conducted. 
        hyperparam_grid_specification : Mapping[str, list]. 
            Default: None. If None, a regression-specific default hyperparameter 
            search is conducted. 
        name : str. 
            Default: None. Determines how the model shows up in the reports. 
            If None, the name is set to be the class name.
        model_random_state : int.
            Default: 42. Random seed for the model.
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

        self._type = type
        if name is None:
            self._name = 'MLPR'
        else:
            self._name = name

        self._estimator = MLPRegressor(random_state=model_random_state)
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


        

