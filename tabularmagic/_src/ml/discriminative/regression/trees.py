from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    BaggingRegressor,
    AdaBoostRegressor,
)
from typing import Mapping, Literal, Iterable
from .base import BaseR, HyperparameterSearcher
import xgboost as xgb
from optuna.distributions import (
    FloatDistribution,
    CategoricalDistribution,
    IntDistribution,
    BaseDistribution,
)
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class TreeR(BaseR):
    """Class for tree-based regression.

    Like all BaseR-derived classes, hyperparameter selection is
    performed automatically during training. The cross validation and
    hyperparameter selection process can be modified by the user.
    """

    def __init__(
        self,
        hyperparam_search_method: Literal["optuna", "grid"] | None = None,
        hyperparam_grid_specification: Mapping[str, Iterable | BaseDistribution]
        | None = None,
        model_random_state: int = 42,
        name: str | None = None,
        **kwargs,
    ):
        """
        Initializes a TreeR object.

        Parameters
        ----------
        hyperparam_search_method : Literal[None, 'grid', 'optuna'].
            Default: None. If None, a regression-specific default hyperparameter
            search is conducted.
        hyperparam_grid_specification : Mapping[str, Iterable | BaseDistribution].
            Default: None. If None, a regression-specific default hyperparameter
            search is conducted.
        model_random_state : int.
            Default: 42. Random seed for the model.
        name : str.
            Default: None. Determines how the model shows up in the reports.
            If None, the name is set to be the class name.
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
        n_trials : int.
            Default: 100. Number of trials for hyperparameter optimization. Only
            used if hyperparam_search_method is 'optuna'.
        """
        super().__init__()

        if name is None:
            self._name = "TreeR"
        else:
            self._name = name

        self._estimator = DecisionTreeRegressor(random_state=model_random_state)
        if (hyperparam_search_method is None) or (
            hyperparam_grid_specification is None
        ):
            hyperparam_search_method = "optuna"
            hyperparam_grid_specification = {
                "max_depth": CategoricalDistribution([3, 6, 12, None]),
                "min_samples_split": FloatDistribution(0.1, 0.5),
                "min_samples_leaf": FloatDistribution(0.1, 0.5),
                "max_features": CategoricalDistribution(["sqrt", "log2", "auto"]),
            }
        self._hyperparam_searcher = HyperparameterSearcher(
            estimator=self._estimator,
            method=hyperparam_search_method,
            hyperparam_grid=hyperparam_grid_specification,
            **kwargs,
        )


class TreeEnsembleR(BaseR):
    """Ensemble of trees regressor. Includes random forest, gradient boosting,
    and bagging.

    Like all BaseRegression-derived classes, hyperparameter selection is
    performed automatically during training. The cross validation and
    hyperparameter selection process can be modified by the user.
    """

    def __init__(
        self,
        type: Literal[
            "random_forest",
            "gradient_boosting",
            "adaboost",
            "bagging",
            "xgboost",
            "xgboostrf",
        ] = "random_forest",
        hyperparam_search_method: Literal["optuna", "grid"] | None = None,
        hyperparam_grid_specification: Mapping[str, Iterable | BaseDistribution]
        | None = None,
        model_random_state: int = 42,
        name: str | None = None,
        **kwargs,
    ):
        """
        Initializes a TreeEnsembleR object.

        Parameters
        ----------
        type : Literal['random_forest', 'gradient_boosting',
                    'adaboost', 'bagging', 'xgboost', 'xgboostrf']
            Default: 'random_forest'. The type of tree ensemble to use.
        hyperparam_search_method : Literal[None, 'grid', 'optuna'].
            Default: None. If None, a regression-specific default hyperparameter
            search is conducted.
        hyperparam_grid_specification : Mapping[str, Iterable | BaseDistribution].
            Default: None. If None, a regression-specific default hyperparameter
            search is conducted.
        model_random_state : int.
            Default: 42. Random seed for the model.
        name : str.
            Default: None. Determines how the model shows up in the reports.
            If None, the name is set to be the class name.
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
        n_trials : int.
            Default: 100. Number of trials for hyperparameter optimization. Only
            used if hyperparam_search_method is 'optuna'.
        """
        super().__init__()

        if name is None:
            self._name = f"TreeEnsembleR({type})"
        else:
            self._name = name

        if type == "random_forest":
            self._estimator = RandomForestRegressor(random_state=model_random_state)
            if (hyperparam_search_method is None) or (
                hyperparam_grid_specification is None
            ):
                hyperparam_search_method = "optuna"
                hyperparam_grid_specification = {
                    "n_estimators": CategoricalDistribution([50, 100, 200, 400]),
                    "min_samples_split": CategoricalDistribution([2, 5, 10]),
                    "min_samples_leaf": CategoricalDistribution([1, 2, 4]),
                    "max_features": CategoricalDistribution(["sqrt", "log2"]),
                    "max_depth": IntDistribution(3, 15, step=2),
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                hyperparam_grid=hyperparam_grid_specification,
                **kwargs,
            )
        elif type == "adaboost":
            self._estimator = AdaBoostRegressor(random_state=model_random_state)
            if (hyperparam_search_method is None) or (
                hyperparam_grid_specification is None
            ):
                hyperparam_search_method = "optuna"
                hyperparam_grid_specification = {
                    "n_estimators": CategoricalDistribution([50, 100, 200]),
                    "learning_rate": FloatDistribution(1e-3, 1e0, log=True),
                    "estimator": CategoricalDistribution(
                        [
                            DecisionTreeRegressor(
                                max_depth=3, random_state=model_random_state
                            ),
                            DecisionTreeRegressor(
                                max_depth=5, random_state=model_random_state
                            ),
                            DecisionTreeRegressor(
                                max_depth=8, random_state=model_random_state
                            ),
                            DecisionTreeRegressor(
                                max_depth=12, random_state=model_random_state
                            ),
                        ]
                    ),
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                hyperparam_grid=hyperparam_grid_specification,
                **kwargs,
            )
        elif type == "bagging":
            self._estimator = BaggingRegressor(random_state=model_random_state)
            if (hyperparam_search_method is None) or (
                hyperparam_grid_specification is None
            ):
                hyperparam_search_method = "optuna"
                hyperparam_grid_specification = {
                    "n_estimators": CategoricalDistribution([50, 100, 200]),
                    "max_samples": FloatDistribution(0.1, 1.0),
                    "max_features": FloatDistribution(0.1, 1.0),
                    "bootstrap": CategoricalDistribution([True, False]),
                    "bootstrap_features": CategoricalDistribution([True, False]),
                    "estimator": CategoricalDistribution(
                        [
                            DecisionTreeRegressor(
                                max_depth=3, random_state=model_random_state
                            ),
                            DecisionTreeRegressor(
                                max_depth=5, random_state=model_random_state
                            ),
                            DecisionTreeRegressor(
                                max_depth=8, random_state=model_random_state
                            ),
                            DecisionTreeRegressor(
                                max_depth=12, random_state=model_random_state
                            ),
                        ]
                    ),
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                hyperparam_grid=hyperparam_grid_specification,
                **kwargs,
            )
        elif type == "gradient_boosting":
            self._estimator = GradientBoostingRegressor(random_state=model_random_state)
            if (hyperparam_search_method is None) or (
                hyperparam_grid_specification is None
            ):
                hyperparam_search_method = "optuna"
                hyperparam_grid_specification = {
                    "n_estimators": CategoricalDistribution([50, 100, 200, 400]),
                    "subsample": FloatDistribution(0.1, 1.0),
                    "min_samples_split": FloatDistribution(0.1, 0.5),
                    "min_samples_leaf": FloatDistribution(0.1, 0.5),
                    "max_depth": IntDistribution(3, 9, step=2),
                    "max_features": CategoricalDistribution(["sqrt", "log2", "auto"]),
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                hyperparam_grid=hyperparam_grid_specification,
                **kwargs,
            )
        elif type == "xgboost":
            self._estimator = xgb.XGBRegressor(random_state=model_random_state)
            if (hyperparam_search_method is None) or (
                hyperparam_grid_specification is None
            ):
                hyperparam_search_method = "optuna"
                hyperparam_grid_specification = {
                    "learning_rate": FloatDistribution(1e-3, 1e0, log=True),
                    "n_estimators": CategoricalDistribution([50, 100, 200]),
                    "max_depth": IntDistribution(3, 9, step=2),
                    "reg_lambda": FloatDistribution(1e-5, 1e0, log=True),
                    "reg_alpha": FloatDistribution(1e-5, 1e0, log=True),
                    "subsample": FloatDistribution(0.6, 1.0),
                    "colsample_bytree": FloatDistribution(0.6, 1.0),
                    "min_child_weight": CategoricalDistribution([1, 3, 5]),
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                hyperparam_grid=hyperparam_grid_specification,
                **kwargs,
            )
        elif type == "xgboostrf":
            self._estimator = xgb.XGBRFRegressor(random_state=model_random_state)
            if (hyperparam_search_method is None) or (
                hyperparam_grid_specification is None
            ):
                hyperparam_search_method = "optuna"
                hyperparam_grid_specification = {
                    "learning_rate": FloatDistribution(1e-3, 1e0, log=True),
                    "max_depth": IntDistribution(3, 9, step=2),
                    "n_estimators": CategoricalDistribution([50, 100, 200]),
                    "min_child_weight": CategoricalDistribution([1, 3, 5]),
                    "subsample": FloatDistribution(0.6, 1.0),
                    "colsample_bytree": FloatDistribution(0.6, 1.0),
                    "reg_lambda": FloatDistribution(1e-5, 1e0, log=True),
                    "reg_alpha": FloatDistribution(1e-5, 1e0, log=True),
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                hyperparam_grid=hyperparam_grid_specification,
                **kwargs,
            )

        else:
            raise ValueError("Invalid input: ensemble_type = " + f'"{type}".')
