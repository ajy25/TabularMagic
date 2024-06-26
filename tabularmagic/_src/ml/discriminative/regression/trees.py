from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    BaggingRegressor,
    AdaBoostRegressor
)
from typing import Mapping, Literal, Iterable
from .base import BaseR, HyperparameterSearcher
import xgboost as xgb


class TreeR(BaseR):
    """Class for tree-based regression.
    
    Like all BaseR-derived classes, hyperparameter selection is 
    performed automatically during training. The cross validation and 
    hyperparameter selection process can be modified by the user. 
    """

    def __init__(self,
                 hyperparam_search_method: str = None, 
                 hyperparam_grid_specification: Mapping[str, Iterable] = None,
                 model_random_state: int = 42,
                 name: str = None, **kwargs):
        """
        Initializes a TreeR object. 

        Parameters
        ----------
        - hyperparam_search_method : str. 
            Default: None. If None, a regression-specific default hyperparameter 
            search is conducted. 
        - hyperparam_grid_specification : Mapping[str, list]. 
            Default: None. If None, a regression-specific default hyperparameter 
            search is conducted. 
        - name : str. 
            Default: None. Determines how the model shows up in the reports. 
            If None, the name is set to be the class name.
        - model_random_state : int.
            Default: 42. Random seed for the model.
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
            self._name = 'TreeR'
        else:
            self._name = name

        self._estimator = DecisionTreeRegressor(random_state=model_random_state)
        if (hyperparam_search_method is None) or \
            (hyperparam_grid_specification is None):
            hyperparam_search_method = 'grid'
            hyperparam_grid_specification = {
                'min_samples_split': [2, 0.1, 0.05],
                'min_samples_leaf': [1, 0.1, 0.05],
                'max_features': ['sqrt', 'log2', None],
            }
        self._hyperparam_searcher = HyperparameterSearcher(
            estimator=self._estimator,
            method=hyperparam_search_method,
            grid=hyperparam_grid_specification,
            **kwargs
        )



class TreeEnsembleR(BaseR):
    """Ensemble of trees regressor. Includes random forest, gradient boosting, 
    and bagging. 
    
    Like all BaseRegression-derived classes, hyperparameter selection is 
    performed automatically during training. The cross validation and 
    hyperparameter selection process can be modified by the user. 
    """

    def __init__(self, 
                 type: Literal['random_forest', 'gradient_boosting', 
                    'adaboost', 'bagging', 'xgboost', 'xgboostrf'] =\
                          'random_forest', 
                 hyperparam_search_method: str = None, 
                 hyperparam_grid_specification: Mapping[str, Iterable] = None,
                 model_random_state: int = 42,
                 name: str = None, **kwargs):
        """
        Initializes a TreeEnsembleR object. 

        Parameters
        ----------
        - type : Literal['random_forest', 'gradient_boosting', 
                    'adaboost', 'bagging', 'xgboost', 'xgboostrf']
            Default: 'random_forest'. The type of tree ensemble to use.
        - hyperparam_search_method : str. 
            Default: None. If None, a Tree-specific default hyperparameter 
            search is conducted. 
        - hyperparam_grid_specification : Mapping[str, Iterable]. 
            Default: None. If None, a Tree-specific default hyperparameter 
            search is conducted. 
        - name : str. 
            Default: None. Determines how the model shows up in the reports. 
            If None, the name is set to be the class name.
        - model_random_state : int.
            Default: 42. Random seed for the model.
        - kwargs : Key word arguments are passed directly into the 
            intialization of the hyperparameter search method. 

        Notable kwargs
        --------------
        - inner_cv : int | BaseCrossValidator. Default is 5.
        - inner_cv_seed : int.
        - n_jobs : int. Number of parallel jobs to run.
        - verbose : int. sklearn verbosity level.
        """
        super().__init__()

        if name is None:
            self._name = f'TreeEnsembleR({type})'
        else:
            self._name = name

        if type == 'random_forest':
            self._estimator = RandomForestRegressor(
                random_state=model_random_state)
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'n_estimators': [50, 100, 200],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2'],
                    'max_depth': [3, 6, 12]
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )
        elif type == 'adaboost':
            self._estimator = AdaBoostRegressor(
                random_state=model_random_state)
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [1.0, 0.1, 0.01],
                    'estimator': [
                        DecisionTreeRegressor(max_depth=3, 
                                              random_state=model_random_state), 
                        DecisionTreeRegressor(max_depth=6, 
                                              random_state=model_random_state),
                        DecisionTreeRegressor(max_depth=12, 
                                              random_state=model_random_state)
                    ]
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )
        elif type == 'bagging':
            self._estimator = BaggingRegressor(
                random_state=model_random_state)
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'n_estimators': [50, 100, 200],
                    'max_samples': [1, 0.5],
                    'max_features': [1, 0.5],
                    'bootstrap': [True, False],
                    'bootstrap_features': [True, False],
                    'estimator': [
                        DecisionTreeRegressor(max_depth=3, 
                                              random_state=model_random_state), 
                        DecisionTreeRegressor(max_depth=6, 
                                              random_state=model_random_state),
                        DecisionTreeRegressor(max_depth=12, 
                                              random_state=model_random_state)
                    ]
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )
        elif type == 'gradient_boosting':
            self._estimator = GradientBoostingRegressor(
                random_state=model_random_state)
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'n_estimators': [50, 100, 200],
                    'subsample': [0.2, 0.5, 1.0],
                    'min_samples_split': [2, 0.1],
                    'min_samples_leaf': [1, 0.1],
                    'max_depth': [5, 10, None],
                    'max_features': ['sqrt', 'log2', None],
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )
        elif type == 'xgboost':
            self._estimator = xgb.XGBRegressor(random_state=model_random_state)
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 12],
                    'lambda': [0, 0.1, 1],
                    'alpha': [0, 0.1, 1],
                    'colsample_bytree': [1, 0.8, 0.5]
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )
        elif type == 'xgboostrf':
            self._estimator = xgb.XGBRFRegressor(
                random_state=model_random_state)
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'max_depth': [3, 6, 12],
                    'n_estimators': [50, 100, 200],
                    'colsample_bynode': [0.5, 0.8, 1.0],
                    'min_child_weight': [1, 3, 5],
                    'subsample': [0.5, 0.8, 1.0],
                    'reg_lambda': [0.1, 1.0, 10.0]
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )

        else:
            raise ValueError('Invalid input: ensemble_type = ' + \
                             f'"{type}".')






