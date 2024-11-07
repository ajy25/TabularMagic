from sklearn.neural_network import MLPClassifier
from typing import Mapping, Iterable, Literal
from .base import BaseC, HyperparameterSearcher
from optuna.distributions import (
    FloatDistribution,
    CategoricalDistribution,
    BaseDistribution,
)
from ....feature_selection import BaseFSC


class MLPC(BaseC):
    """Multi-layer Perceptron classifier.

    Hyperparameter optimization is performed automatically during training.
    The hyperparameter search process can be modified by the user.
    """

    def __init__(
        self,
        hyperparam_search_method: Literal["optuna", "grid"] | None = None,
        hyperparam_search_space: (
            Mapping[str, Iterable | BaseDistribution] | None
        ) = None,
        feature_selectors: list[BaseFSC] | None = None,
        max_n_features: int | None = 10,
        model_random_state: int = 42,
        name: str | None = None,
        **kwargs,
    ):
        """
        Initializes an MLPC object.

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
            Default: None. Determines how the model shows up in the report.
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

        self._feature_selectors = feature_selectors
        self._max_n_features = max_n_features

        self._type = type
        if name is None:
            self._name = "MLPC"
        else:
            self._name = name

        self._best_estimator = MLPClassifier(random_state=model_random_state)
        if (hyperparam_search_method is None) or (hyperparam_search_space is None):
            hyperparam_search_method = "optuna"
            hyperparam_search_space = {
                "hidden_layer_sizes": CategoricalDistribution(
                    [
                        (50,),
                        (100,),
                        (
                            50,
                            25,
                        ),
                        (
                            50,
                            50,
                        ),
                        (
                            100,
                            50,
                        ),
                        (
                            100,
                            50,
                            25,
                        ),
                    ]
                ),
                "activation": CategoricalDistribution(["relu", "tanh"]),
                "alpha": FloatDistribution(1e-5, 1e0, log=True),
                "learning_rate": CategoricalDistribution(["constant", "adaptive"]),
                "learning_rate_init": FloatDistribution(1e-5, 1e-1, log=True),
            }
        self._hyperparam_searcher = HyperparameterSearcher(
            estimator=self._best_estimator,
            method=hyperparam_search_method,
            hyperparam_grid=hyperparam_search_space,
            estimator_name=self._name,
            **kwargs,
        )

        self._validate_inputs()
