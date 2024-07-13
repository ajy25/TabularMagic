import numpy as np
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    BaseCrossValidator,
)
from typing import Literal, Mapping, Iterable
from sklearn.base import BaseEstimator
from sklearn.utils._testing import ignore_warnings
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
from ...display.print_utils import print_wrapped


class BaseDiscriminativeModel:
    """Skeletal class for typing assistance of BaseR and BaseC.

    BaseR and BaseC extend BaseDiscriminativeModel.
    BaseDiscriminativeModel has no funtionality beyond providing
    typing assistance elsewhere.
    """

    def __init__(self):
        self._id = "BaseDiscriminativeModel"

    def __str__(self):
        return self._id


class HyperparameterSearcher:
    """Class for searching for hyperparameters."""

    def __init__(
        self,
        estimator: BaseEstimator,
        method: Literal["optuna", "grid"],
        hyperparam_grid: Mapping[str, Iterable],
        inner_cv: int | BaseCrossValidator = 5,
        inner_cv_seed: int = 42,
        **kwargs
    ):
        """Initializes a HyperparameterSearch object.

        Parameters
        ----------
        - estimator : sklearn.base.BaseEstimator.
        - method : str.
            Must be an element in ['optuna', 'grid'].
        - hyperparam_grid : dict.
            Specification of the set/distribution of hyperparameters to
            search through.
        - inner_cv : int | BaseCrossValidator.
            Default: 5-fold cross validation.
        - inner_cv_seed : int.
            Default: 42.
        - kwargs.
            Key word arguments are passed directly into the intialization of the
            hyperparameter search method.
        """

        self._best_estimator = None
        if isinstance(inner_cv, int):
            self.inner_cv = KFold(
                n_splits=inner_cv, random_state=inner_cv_seed, shuffle=True
            )
        elif isinstance(inner_cv, BaseCrossValidator):
            self.inner_cv = inner_cv
        else:
            raise ValueError("Invalid input: inner_cv.")

        if method == "optuna":
            if "n_trials" not in kwargs:
                kwargs["n_trials"] = 100
            if "verbose" not in kwargs:
                kwargs["verbose"] = 0
                optuna.logging.set_verbosity(optuna.logging.WARNING)
            else:
                if kwargs["verbose"] not in [0, 1, 2]:
                    raise ValueError("Invalid input: verbose.")
                else:
                    if kwargs["verbose"] == 2:
                        optuna.logging.set_verbosity(optuna.logging.DEBUG)
                    elif kwargs["verbose"] == 1:
                        optuna.logging.set_verbosity(optuna.logging.INFO)
                    else:
                        optuna.logging.set_verbosity(optuna.logging.WARNING)
            self._searcher = optuna.integration.OptunaSearchCV(
                estimator=estimator,
                param_distributions=hyperparam_grid,
                cv=self.inner_cv,
                random_state=inner_cv_seed + 1,
                **kwargs
            )

        elif method == "grid":
            self._searcher = GridSearchCV(
                estimator=estimator,
                param_grid=hyperparam_grid,
                cv=self.inner_cv,
                **kwargs
            )
        else:
            raise ValueError("Invalid input: method.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
        """Cross validation search of optimal hyperparameters. Idenfities
        best estimator.

        Parameters
        ----------
        - X : np.ndarray ~ (sample_size, n_predictors).
        - y : np.ndarray ~ (sample_size).

        Returns
        -------
        - best_estimator : BaseEstimator.
        """
        ignore_warnings(self._searcher.fit)(X, y)
        self._best_estimator = self._searcher.best_estimator_
        self._best_params = self._searcher.best_params_
        return self._best_estimator

    def best_estimator(self) -> BaseEstimator:
        """Returns the best estimator.

        Returns
        -------
        - BaseEstimator
        """
        return self._best_estimator

    def best_params(self) -> Mapping:
        """Returns the best parameters.

        Returns
        -------
        - Mapping
        """
        return self._best_params
