import numpy as np
from .base_regression import BaseRegression, HyperparameterSearcher
import xgboost as xgb


class Tree(BaseRegression):
    """A simple decision tree regressor. 
    
    Like all BaseRegression-derived classes, hyperparameter selection is 
    performed automatically during training. The cross validation and 
    hyperparameter selection process can be modified by the user. 
    """

    def __init__(self, X: np.ndarray = None, y: np.ndarray = None, 
                 random_state: int = 42, 
                 hyperparam_search_method: str = None, 
                 hyperparam_grid_specification: Mapping[str, Iterable] = None,
                 nickname: str = None, **kwargs):
        """
        Initializes a TreeRegression object. 

        Parameters
        ----------
        - X : np.ndarray ~ (n_samples, n_regressors).
            Default: None. Matrix of predictor variables. 
        - y : np.ndarray ~ (n_samples).
            Default: None. Dependent variable vector. 
        - random_state : int. 
            Default: 42.
        - hyperparam_search_method : str. 
            Default: None. If None, a Tree-specific default hyperparameter 
            search is conducted. 
        - hyperparam_grid_specification : Mapping[str, Iterable]. 
            Default: None. If None, a Tree-specific default hyperparameter 
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
            self.nickname = 'Tree'
        else:
            self.nickname = nickname

        self.random_state = random_state
        self.estimator = DecisionTreeRegressor(random_state=self.random_state)
        if (hyperparam_search_method is None) or \
            (hyperparam_grid_specification is None):
            hyperparam_search_method = 'grid'
            hyperparam_grid_specification = {
                'min_samples_split': [2, 0.1, 0.05],
                'min_samples_leaf': [1, 0.1, 0.05],
                'max_features': ['sqrt', 'log2', None],
            }
        self._hyperparam_searcher = HyperparameterSearcher(
            estimator=self.estimator,
            method=hyperparam_search_method,
            grid=hyperparam_grid_specification,
            **kwargs
        )

    def __str__(self):
        return self.nickname

