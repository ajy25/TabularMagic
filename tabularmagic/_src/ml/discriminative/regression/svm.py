from sklearn.svm import SVR
from typing import Mapping, Iterable, Literal
from .base import BaseR, HyperparameterSearcher
from optuna.distributions import (
    FloatDistribution,
    IntDistribution,
    CategoricalDistribution,
    BaseDistribution,
)
from ....feature_selection import BaseFSR


class SVMR(BaseR):
    """Class for support vector machine regression.

    Like all BaseR-derived classes, hyperparameter selection is
    performed automatically during training. The cross validation and
    hyperparameter selection process can be modified by the user.
    """

    def __init__(
        self,
        type: Literal["linear", "poly", "rbf"] = "rbf",
        hyperparam_search_method: Literal["optuna", "grid"] | None = None,
        hyperparam_search_space: (
            Mapping[str, Iterable | BaseDistribution] | None
        ) = None,
        feature_selectors: list[BaseFSR] | None = None,
        max_n_features: int = 10,
        name: str | None = None,
        **kwargs,
    ):
        """
        Initializes a SVMR object.

        Parameters
        ----------
        type : Literal['linear', 'poly', 'rbf'].
            Default: 'rbf'. The type of kernel to use.
        hyperparam_search_method : Literal[None, 'grid', 'optuna'].
            Default: None. If None, a regression-specific default hyperparameter
            search is conducted.
        hyperparam_search_space : Mapping[str, Iterable | BaseDistribution].
            Default: None. If None, a regression-specific default hyperparameter
            search is conducted.
        feature_selectors : list[BaseFSR].
            Default: None. If not None, specifies the feature selectors for the
            VotingSelectionReport.
        max_n_features : int.
            Default: 10. Maximum number of features to select. Only useful if
            feature_selectors is not None.
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
            self._name = f"SVMR({type})"
        else:
            self._name = name

        self._estimator = SVR(kernel=type)
        self._feature_selectors = feature_selectors
        self._max_n_features = max_n_features

        if (hyperparam_search_method is None) or (hyperparam_search_space is None):
            hyperparam_search_method = "optuna"

            if type == "linear":
                hyperparam_search_space = {
                    "C": FloatDistribution(1e-2, 1e2, log=True),
                    "epsilon": FloatDistribution(1e-3, 1e0, log=True),
                }
            elif type == "poly":
                hyperparam_search_space = {
                    "C": FloatDistribution(1e-2, 1e2, log=True),
                    "epsilon": FloatDistribution(1e-3, 1e0, log=True),
                    "degree": IntDistribution(2, 5),
                    "coef0": IntDistribution(0, 10),
                    "gamma": CategoricalDistribution(["scale", "auto"]),
                }
            elif type == "rbf":
                hyperparam_search_space = {
                    "C": FloatDistribution(1e-2, 1e2, log=True),
                    "epsilon": FloatDistribution(1e-3, 1e0, log=True),
                    "gamma": CategoricalDistribution(["scale", "auto"]),
                }

        self._hyperparam_searcher = HyperparameterSearcher(
            estimator=self._estimator,
            method=hyperparam_search_method,
            hyperparam_grid=hyperparam_search_space,
            estimator_name=self._name,
            **kwargs,
        )
