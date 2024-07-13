from sklearn.neural_network import MLPClassifier
from typing import Mapping, Iterable, Literal
from .base import BaseC, HyperparameterSearcher
from optuna.distributions import (
    FloatDistribution,
    CategoricalDistribution,
    BaseDistribution,
)


class MLPC(BaseC):
    """Multi-layer Perceptron classifier.

    Like all BaseC-derived classes, hyperparameter selection is
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
        **kwargs
    ):
        """
        Initializes an MLPC object.

        Parameters
        ----------
        hyperparam_search_method : Literal[None, 'grid', 'optuna'].
            Default: None. If None, a classification-specific default hyperparameter
            search is conducted.
        hyperparam_grid_specification : Mapping[str, Iterable | BaseDistribution].
            Default: None. If None, a classification-specific default hyperparameter
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
        n_trials : int.
            Default: 100. Number of trials for hyperparameter optimization. Only
            used if hyperparam_search_method is 'optuna'.
        """
        super().__init__()

        self._type = type
        if name is None:
            self._name = "MLPC"
        else:
            self._name = name

        self._estimator = MLPClassifier(random_state=model_random_state)
        if (hyperparam_search_method is None) or (
            hyperparam_grid_specification is None
        ):
            hyperparam_search_method = "optuna"
            hyperparam_grid_specification = {
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
            estimator=self._estimator,
            method=hyperparam_search_method,
            hyperparam_grid=hyperparam_grid_specification,
            **kwargs
        )
