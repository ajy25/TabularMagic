from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
import xgboost as xgb
from typing import Mapping, Literal, Iterable
from .base import BaseC, HyperparameterSearcher
from optuna.distributions import (
    FloatDistribution,
    CategoricalDistribution,
    IntDistribution,
    BaseDistribution,
)
from ....feature_selection import BaseFSC


class TreeC(BaseC):
    """Decision tree classifier.

    Hyperparameter optimization is performed automatically during training. 
    The hyperparameter search process can be modified by the user.
    """

    def __init__(
        self,
        hyperparam_search_method: Literal["optuna", "grid"] | None = None,
        hyperparam_search_space: (
            Mapping[str, Iterable | BaseDistribution] | None
        ) = None,
        model_random_state: int = 42,
        feature_selectors: list[BaseFSC] | None = None,
        max_n_features: int | None = None,
        name: str | None = None,
        **kwargs,
    ):
        """
        Initializes a TreeC object.

        Parameters
        ----------
        hyperparam_search_method : Literal[None, 'grid', 'optuna']
            Default: None. If None, a model-specific default hyperparameter search
            is conducted.

        hyperparam_search_space : Mapping[str, Iterable | BaseDistribution]
            Default: None. If None, a model-specific default hyperparameter search
            is conducted.

        feature_selectors : list[BaseFSC]
            Default: None. If not None, specifies the feature selectors for the
            VotingSelectionReport.

        max_n_features : int | None
            Default: None.
            Only useful if feature_selectors is not None.
            If None, then all features with at least 50% support are selected.

        model_random_state : int
            Default: 42. Random seed for the model.

        name : str
            Default: None. Determines how the model shows up in the reports.
            If None, the name is set to be the class name.

        **kwargs : dict
            Key word arguments are passed directly into the intialization of the 
            HyperparameterSearcher class. See below for options.

            inner_cv : int | BaseCrossValidator
                Default: 5. Number of inner cross validation folds. Inner
                cross validation is used for hyperparameter optimization.

            inner_cv_seed : int
                Default: 42. Random seed for inner cross validation.

            n_jobs : int
                Default: 1. Number of parallel jobs to run.

            verbose : int
                Default: 0. Sets the sklearn verbosity level for the sklearn estimator.
                2 is the most verbose.

            n_trials : int
                Default: 100. Number of trials for hyperparameter optimization. Only
                used if hyperparam_search_method is 'optuna'.
        """
        super().__init__()

        if name is None:
            self._name = "TreeC"
        else:
            self._name = name

        self._estimator = DecisionTreeClassifier(random_state=model_random_state)
        self._feature_selectors = feature_selectors
        self._max_n_features = max_n_features

        if (hyperparam_search_method is None) or (hyperparam_search_space is None):
            hyperparam_search_method = "optuna"
            hyperparam_search_space = {
                "max_depth": CategoricalDistribution([3, 6, 12, None]),
                "min_samples_split": FloatDistribution(0.1, 0.5),
                "min_samples_leaf": FloatDistribution(0.1, 0.5),
                "max_features": CategoricalDistribution(["sqrt", "log2", "auto"]),
            }
        self._hyperparam_searcher = HyperparameterSearcher(
            estimator=self._estimator,
            method=hyperparam_search_method,
            hyperparam_grid=hyperparam_search_space,
            estimator_name=self._name,
            **kwargs,
        )


class TreeEnsembleC(BaseC):
    """Tree ensemble classifier.

    Hyperparameter optimization is performed automatically during training. 
    The hyperparameter search process can be modified by the user.
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
        hyperparam_search_space: (
            Mapping[str, Iterable | BaseDistribution] | None
        ) = None,
        feature_selectors: list[BaseFSC] | None = None,
        max_n_features: int | None = None,
        model_random_state: int = 42,
        name: str | None = None,
        **kwargs,
    ):
        """
        Initializes a TreeEnsembleC object.

        Parameters
        ----------
        type : Literal['random_forest', 'gradient_boosting',
                    'adaboost', 'bagging', 'xgboost', 'xgboostrf']
            Default: 'random_forest'. The type of tree ensemble to use.

        hyperparam_search_method : Literal[None, 'grid', 'optuna']
            Default: None. If None, a model-specific default hyperparameter search
            is conducted.

        hyperparam_search_space : Mapping[str, Iterable | BaseDistribution]
            Default: None. If None, a model-specific default hyperparameter search
            is conducted.

        feature_selectors : list[BaseFSC]
            Default: None. If not None, specifies the feature selectors for the
            VotingSelectionReport.

        max_n_features : int | None
            Default: None.
            Only useful if feature_selectors is not None.
            If None, then all features with at least 50% support are selected.

        model_random_state : int
            Default: 42. Random seed for the model.

        name : str
            Default: None. Determines how the model shows up in the reports.
            If None, the name is set to be the class name.

        **kwargs : dict
            Key word arguments are passed directly into the intialization of the 
            HyperparameterSearcher class. See below for options.

            inner_cv : int | BaseCrossValidator
                Default: 5. Number of inner cross validation folds. Inner
                cross validation is used for hyperparameter optimization.

            inner_cv_seed : int
                Default: 42. Random seed for inner cross validation.

            n_jobs : int
                Default: 1. Number of parallel jobs to run.

            verbose : int
                Default: 0. Sets the sklearn verbosity level for the sklearn estimator.
                2 is the most verbose.

            n_trials : int
                Default: 100. Number of trials for hyperparameter optimization. Only
                used if hyperparam_search_method is 'optuna'.
        """
        super().__init__()
        self._dropfirst = True

        if name is None:
            self._name = f"TreeEnsembleC({type})"
        else:
            self._name = name

        self._feature_selectors = feature_selectors
        self._max_n_features = max_n_features

        if type == "random_forest":
            self._estimator = RandomForestClassifier(random_state=model_random_state)
            if (hyperparam_search_method is None) or (hyperparam_search_space is None):
                hyperparam_search_method = "optuna"
                hyperparam_search_space = {
                    "n_estimators": CategoricalDistribution([50, 100, 200, 400]),
                    "min_samples_split": CategoricalDistribution([2, 5, 10]),
                    "min_samples_leaf": CategoricalDistribution([1, 2, 4]),
                    "max_features": CategoricalDistribution(["sqrt", "log2"]),
                    "max_depth": IntDistribution(3, 15, step=2),
                }

        elif type == "adaboost":
            self._estimator = AdaBoostClassifier(random_state=model_random_state)
            if (hyperparam_search_method is None) or (hyperparam_search_space is None):
                hyperparam_search_method = "optuna"
                hyperparam_search_space = {
                    "n_estimators": CategoricalDistribution([50, 100, 200]),
                    "learning_rate": FloatDistribution(1e-3, 1e0, log=True),
                    "estimator": CategoricalDistribution(
                        [
                            DecisionTreeClassifier(
                                max_depth=3, random_state=model_random_state
                            ),
                            DecisionTreeClassifier(
                                max_depth=5, random_state=model_random_state
                            ),
                            DecisionTreeClassifier(
                                max_depth=8, random_state=model_random_state
                            ),
                            DecisionTreeClassifier(
                                max_depth=12, random_state=model_random_state
                            ),
                        ]
                    ),
                }

        elif type == "bagging":
            self._estimator = BaggingClassifier(random_state=model_random_state)
            if (hyperparam_search_method is None) or (hyperparam_search_space is None):
                hyperparam_search_method = "grid"
                hyperparam_search_space = {
                    "n_estimators": CategoricalDistribution([50, 100, 200]),
                    "max_samples": FloatDistribution(0.1, 1.0),
                    "max_features": FloatDistribution(0.1, 1.0),
                    "bootstrap": CategoricalDistribution([True, False]),
                    "bootstrap_features": CategoricalDistribution([True, False]),
                    "estimator": CategoricalDistribution(
                        [
                            DecisionTreeClassifier(
                                max_depth=3, random_state=model_random_state
                            ),
                            DecisionTreeClassifier(
                                max_depth=5, random_state=model_random_state
                            ),
                            DecisionTreeClassifier(
                                max_depth=8, random_state=model_random_state
                            ),
                            DecisionTreeClassifier(
                                max_depth=12, random_state=model_random_state
                            ),
                        ]
                    ),
                }

        elif type == "gradient_boosting":
            self._estimator = GradientBoostingClassifier(
                random_state=model_random_state
            )
            if (hyperparam_search_method is None) or (hyperparam_search_space is None):
                hyperparam_search_method = "optuna"
                hyperparam_search_space = {
                    "n_estimators": CategoricalDistribution([50, 100, 200, 400]),
                    "subsample": FloatDistribution(0.1, 1.0),
                    "min_samples_split": FloatDistribution(0.1, 0.5),
                    "min_samples_leaf": FloatDistribution(0.1, 0.5),
                    "max_depth": IntDistribution(3, 9, step=2),
                    "max_features": CategoricalDistribution(["sqrt", "log2", "auto"]),
                }

        elif type == "xgboost":
            self._estimator = xgb.XGBClassifier(random_state=model_random_state)
            if (hyperparam_search_method is None) or (hyperparam_search_space is None):
                hyperparam_search_method = "optuna"
                hyperparam_search_space = {
                    "learning_rate": FloatDistribution(1e-3, 1e0, log=True),
                    "n_estimators": CategoricalDistribution([50, 100, 200]),
                    "max_depth": IntDistribution(3, 9, step=2),
                    "reg_lambda": FloatDistribution(1e-5, 1e0, log=True),
                    "reg_alpha": FloatDistribution(1e-5, 1e0, log=True),
                    "subsample": FloatDistribution(0.5, 1.0),
                    "colsample_bytree": FloatDistribution(0.5, 1.0),
                    "min_child_weight": CategoricalDistribution([1, 3, 5]),
                }

        elif type == "xgboostrf":
            self._estimator = xgb.XGBRFClassifier(random_state=model_random_state)
            if (hyperparam_search_method is None) or (hyperparam_search_space is None):
                hyperparam_search_method = "optuna"
                hyperparam_search_space = {
                    "learning_rate": FloatDistribution(1e-3, 1e0, log=True),
                    "max_depth": IntDistribution(3, 9, step=2),
                    "n_estimators": CategoricalDistribution([50, 100, 200]),
                    "min_child_weight": CategoricalDistribution([1, 3, 5]),
                    "subsample": FloatDistribution(0.6, 1.0),
                    "colsample_bytree": FloatDistribution(0.6, 1.0),
                    "reg_lambda": FloatDistribution(1e-5, 1e0, log=True),
                    "reg_alpha": FloatDistribution(1e-5, 1e0, log=True),
                }

        else:
            raise ValueError("Invalid value for type")

        self._hyperparam_searcher = HyperparameterSearcher(
            estimator=self._estimator,
            method=hyperparam_search_method,
            hyperparam_grid=hyperparam_search_space,
            estimator_name=self._name,
            **kwargs,
        )
