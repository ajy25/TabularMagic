from sklearn.linear_model import LogisticRegression
from typing import Mapping, Literal, Iterable


from .base import BaseC, HyperparameterSearcher
from optuna.distributions import (
    FloatDistribution,
    BaseDistribution,
)
from ....feature_selection import BaseFSC


class LinearC(BaseC):
    """Logistic Regression classifier.

    Hyperparameter tuning is performed automatically during training.
    """

    def __init__(
        self,
        type: Literal["no_penalty", "l1", "l2", "elasticnet"] = "l2",
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
        Initializes a LinearC object.

        Parameters
        ----------
        type: Literal['no_penalty', 'l1', 'l2', 'elasticnet']
            Default: 'l2'. Specifies the type of logistic regression penalty.

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

        **kwargs : Key word arguments are passed directly into the
            intialization of the HyperparameterSearcher class. In particular,
            inner_cv and inner_cv_seed can be set via kwargs.

        **kwargs
        --------------
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
        self._feature_selectors = feature_selectors
        self._max_n_features = max_n_features
        if name is None:
            self._name = f"LinearC({type})"
        else:
            self._name = name

        if type == "no_penalty":
            self._estimator = LogisticRegression(
                penalty=None, random_state=model_random_state
            )
            if (hyperparam_search_method is None) or (hyperparam_search_space is None):
                hyperparam_search_method = "grid"
                hyperparam_search_space = {"fit_intercept": [True]}

        elif type == "l1":
            self._estimator = LogisticRegression(
                penalty="l1", random_state=model_random_state, solver="liblinear"
            )
            if (hyperparam_search_method is None) or (hyperparam_search_space is None):
                hyperparam_search_method = "optuna"
                hyperparam_search_space = {"C": FloatDistribution(1e-2, 1e2, log=True)}

        elif type == "l2":
            self._estimator = LogisticRegression(
                penalty="l2", random_state=model_random_state
            )
            if (hyperparam_search_method is None) or (hyperparam_search_space is None):
                hyperparam_search_method = "optuna"
                hyperparam_search_space = {"C": FloatDistribution(1e-2, 1e2, log=True)}

        elif type == "elasticnet":
            self._estimator = LogisticRegression(
                penalty="elasticnet", random_state=model_random_state, solver="saga"
            )
            if (hyperparam_search_method is None) or (hyperparam_search_space is None):
                hyperparam_search_method = "optuna"
                hyperparam_search_space = {
                    "C": FloatDistribution(1e-2, 1e2, log=True),
                    "l1_ratio": FloatDistribution(0.0, 1.0),
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
