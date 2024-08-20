import numpy as np
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    BaseCrossValidator,
)
from typing import Literal, Mapping, Iterable, Any
from sklearn.base import BaseEstimator
from sklearn.utils._testing import ignore_warnings
import optuna
import warnings
from ...display.print_utils import print_wrapped, fill_ignore_format, quote_and_color




class BasePredictModel:
    """Base class for typing assistance of BaseR and BaseC.

    BaseR and BaseC extend BasePredictModel.
    BasePredictModel has no funtionality beyond providing
    typing assistance elsewhere.
    """

    def __init__(self):
        self._id = "BasePredictModel"

    def __str__(self):
        return self._id


class HyperparameterSearcher:
    """Class for searching for hyperparameters."""

    def __init__(
        self,
        estimator: BaseEstimator,
        method: Literal["optuna", "grid"],
        hyperparam_grid: Mapping[str, Iterable | optuna.distributions.BaseDistribution],
        inner_cv: int | BaseCrossValidator = 5,
        inner_cv_seed: int = 42,
        estimator_name: str | None = None,
        **kwargs: dict,
    ):
        """Initializes a HyperparameterSearch object.

        Parameters
        ----------
        estimator : BaseEstimator

        method : str
            Must be an element in ['optuna', 'grid'].

        hyperparam_grid : Mapping[str, Iterable | BaseDistribution]
            Specification of the set/distribution of hyperparameters to
            search through.

        inner_cv : int | BaseCrossValidator
            Default: 5-fold cross validation.

        inner_cv_seed : int
            Default: 42.

        estimator_name : str
            Default: None. Name of the estimator.

        **kwargs : dict
            Key word arguments are passed directly into the intialization of the
            hyperparameter search method.
        """

        self._best_estimator = None
        if estimator_name is None:
            self._estimator_name = estimator.__class__.__name__
        else:
            self._estimator_name = estimator_name

        if isinstance(inner_cv, int):
            self.inner_cv = KFold(
                n_splits=inner_cv, random_state=inner_cv_seed, shuffle=True
            )
        elif isinstance(inner_cv, BaseCrossValidator):
            self.inner_cv = inner_cv
        else:
            raise ValueError("Invalid input: inner_cv.")

        self._fit_message = ""

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
            self._fit_message = (
                "Search method: OptunaSearchCV "
                f'({kwargs["n_trials"]} trials, '
                f'{self.inner_cv.get_n_splits() * kwargs["n_trials"]} total fits).'
            )

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=optuna.exceptions.ExperimentalWarning
                )
                warnings.filterwarnings("ignore", category=UserWarning)
                self._searcher = optuna.integration.OptunaSearchCV(
                    estimator=estimator,
                    param_distributions=hyperparam_grid,
                    cv=self.inner_cv,
                    enable_pruning=False,
                    random_state=inner_cv_seed + 1,
                    **kwargs,
                )

        elif method == "grid":
            n_fits = 1
            for key in hyperparam_grid.keys():
                if not isinstance(hyperparam_grid[key], Iterable):
                    raise ValueError("Invalid input: hyperparam_grid.")
                n_fits *= len(hyperparam_grid[key])

            self._fit_message = f"Search method: GridSearchCV ({n_fits} fits per fold, "
            self._fit_message += f"{n_fits * self.inner_cv.get_n_splits()} total fits)."
            self._searcher = GridSearchCV(
                estimator=estimator,
                param_grid=hyperparam_grid,
                cv=self.inner_cv,
                **kwargs,
            )
        else:
            raise ValueError("Invalid input: method. Must be 'optuna' or 'grid'.")

        self._fit_message = fill_ignore_format(self._fit_message)

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> BaseEstimator:
        """Cross validation search of optimal hyperparameters. Idenfities
        best estimator.

        Parameters
        ----------
        X : np.ndarray ~ (n_obs, n_predictors)
            NumPy array of predictor variables' values.

        y : np.ndarray ~ (n_obs)
            NumPy array of target variable values.

        verbose : bool
            Default: False. If True, prints progress.

        Returns
        -------
        BaseEstimator
            The best estimator.
        """
        if verbose:
            print_wrapped(
                f"Fitting {quote_and_color(self._estimator_name)}. "
                + self._fit_message,
                type="PROGRESS",
            )
        ignore_warnings(self._searcher.fit)(X, y)
        self._best_estimator = self._searcher.best_estimator_
        self._best_params = self._searcher.best_params_
        return self._best_estimator

    def best_estimator(self) -> BaseEstimator:
        """Returns the best estimator.

        Returns
        -------
        BaseEstimator
            The best estimator.
        """
        return self._best_estimator

    def best_params(self) -> Any:
        """Returns the best parameters.

        Returns
        -------
        Any
            The best set of hyperparameters.
        """
        return self._best_params
