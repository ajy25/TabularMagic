import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    BaggingRegressor,
    AdaBoostRegressor
)
from typing import Mapping, Literal, Iterable
from .base_regression import BaseRegression, HyperparameterSearcher
import xgboost as xgb


class TreeR(BaseRegression):
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
        Initializes a TreeR object. 

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
            intialization of the HyperparameterSearcher class. In particular, 
            inner_cv and inner_random_state can be set via kwargs. 

        Returns
        -------
        - None
        """
        super().__init__(X, y)

        if nickname is None:
            self.nickname = 'TreeR'
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


class TreeEnsembleR(BaseRegression):
    """Ensemble of trees regressor. Includes random forest, gradient boosting, 
    and bagging. 
    
    Like all BaseRegression-derived classes, hyperparameter selection is 
    performed automatically during training. The cross validation and 
    hyperparameter selection process can be modified by the user. 
    """

    def __init__(self, X: np.ndarray = None, y: np.ndarray = None, 
                 ensemble_type: Literal['random_forest', 'gradient_boosting', 
                    'adaboost', 'bagging', 'xgboost'] = 'random_forest', 
                 random_state: int = 42, hyperparam_search_method: str = None, 
                 hyperparam_grid_specification: Mapping[str, Iterable] = None,
                 nickname: str = None, **kwargs):
        """
        Initializes a TreeEnsembleR object. 

        Parameters
        ----------
        - X : np.ndarray ~ (n_samples, n_regressors).
            Default: None. Matrix of predictor variables. 
        - y : np.ndarray ~ (n_samples).
            Default: None. Dependent variable vector. 
        - ensemble_type: Literal['random_forest', 'gradient_boosting', 
                    'adaboost', 'bagging', 'xgboost']
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
        self.random_state = random_state
        self.ensemble_type = ensemble_type

        if nickname is None:
            self.nickname = f'TreeEnsembleR({ensemble_type})'
        else:
            self.nickname = nickname

        if ensemble_type == 'random_forest':
            self.estimator = RandomForestRegressor(
                random_state=self.random_state)
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'n_estimators': [50, 100, 200],
                    'min_samples_split': [2, 0.1],
                    'min_samples_leaf': [1, 0.1],
                    'max_features': ['sqrt', 'log2', None],
                    'max_depth': [5, 10, None]
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self.estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )
        elif ensemble_type == 'adaboost':
            self.estimator = AdaBoostRegressor(
                random_state=self.random_state)
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'n_estimators': [25, 50, 100],
                    'learning_rate': [0.01, 0.001],
                    'estimator': [
                        DecisionTreeRegressor(max_depth=5), 
                        DecisionTreeRegressor(max_depth=10),
                        DecisionTreeRegressor(max_depth=None)
                    ]
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self.estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )
        elif ensemble_type == 'bagging':
            self.estimator = BaggingRegressor(
                random_state=self.random_state)
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'n_estimators': [25, 50, 100],
                    'max_samples': [1, 0.5],
                    'max_features': [1, 0.5],
                    'bootstrap': [True, False],
                    'bootstrap_features': [True, False],
                    'estimator': [
                        DecisionTreeRegressor(max_depth=5), 
                        DecisionTreeRegressor(max_depth=10),
                        DecisionTreeRegressor(max_depth=None)
                    ]
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self.estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )
        elif ensemble_type == 'gradient_boosting':
            self.estimator = GradientBoostingRegressor(
                random_state=self.random_state)
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'n_estimators': [25, 50, 100],
                    'subsample': [0.2, 0.5, 1.0],
                    'min_samples_split': [2, 0.1],
                    'min_samples_leaf': [1, 0.1],
                    'max_depth': [5, 10, None],
                    'max_features': ['sqrt', 'log2', None],
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self.estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )
        elif ensemble_type == 'xgboost':
            self.estimator = xgb.XGBRegressor()
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'learning_rate': [0.01, 0.001],
                    'n_estimators': [25, 50, 100],
                    'max_depth': [3, 6, 12],
                    'lambda': [0, 0.1, 1],
                    'alpha': [0, 0.1, 1],
                    'colsample_bytree': [1, 0.8, 0.5]
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self.estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )
        else:
            raise ValueError(f'Invalid input: ensemble_type = ' + \
                             '"{ensemble_type}".')






