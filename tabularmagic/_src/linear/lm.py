import statsmodels.api as sm
import numpy as np
import pandas as pd
from typing import Literal
from ..metrics.regression_scoring import RegressionScorer
from ..data.datahandler import DataEmitter
from ..utils import ensure_arg_list_uniqueness


def score_ols_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    feature_list,
    metric: Literal["aic", "bic"],
) -> float:
    """Calculates the AIC or BIC score for a given model.

    Parameters
    ----------
    X_train : pd.DataFrame
        The training data.
    y_train : pd.Series
        The training target.
    feature_list : list[str]
        The list of features to include in the model.
    metric : str
        The metric to use for scoring. Either 'aic' or 'bic'.

    Returns
    -------
    float
        The AIC or BIC score for the model.
    """
    if len(feature_list) == 0:
        return np.inf

    subset_X_train = X_train[feature_list]
    new_model = sm.OLS(y_train, subset_X_train).fit(cov_type="HC3")
    if metric == "aic":
        score = new_model.aic
    elif metric == "bic":
        score = new_model.bic
    return score


class OLSLinearModel:
    """Statsmodels OLS wrapper."""

    def __init__(self, name: str | None = None):
        """
        Initializes a OrdinaryLeastSquares object. Regresses y on X.

        Parameters
        ----------
        name : str
            Default: None. Determines how the model shows up in the reports.
            If None, the name is set to be the class name.
        """
        self.estimator = None
        self._name = name
        if self._name is None:
            self._name = "Ordinary Least Squares"

    def specify_data(self, dataemitter: DataEmitter):
        """Adds a DataEmitter object to the model.

        Parameters
        ----------
        dataemitter : DataEmitter
            The DataEmitter containing all the data.
        """
        self._dataemitter = dataemitter

    def fit(self):
        """Fits the model based on the data specified."""
        y_scaler = self._dataemitter.y_scaler()

        X_train, y_train = self._dataemitter.emit_train_Xy()
        n_predictors = X_train.shape[1]
        X_train = sm.add_constant(X_train)
        self.estimator = sm.OLS(y_train, X_train).fit(cov_type="HC3")

        y_pred_train = self.estimator.predict(X_train).to_numpy()
        if y_scaler is not None:
            y_pred_train = y_scaler.inverse_transform(y_pred_train)
            y_train = y_scaler.inverse_transform(y_train)

        self.train_scorer = RegressionScorer(
            y_pred=y_pred_train,
            y_true=y_train.to_numpy(),
            n_predictors=n_predictors,
            name=self._name,
        )

        X_test, y_test = self._dataemitter.emit_test_Xy()
        X_test = sm.add_constant(X_test)

        y_pred_test = self.estimator.predict(X_test).to_numpy()
        if y_scaler is not None:
            y_pred_test = y_scaler.inverse_transform(y_pred_test)
            y_test = y_scaler.inverse_transform(y_test)

        self.test_scorer = RegressionScorer(
            y_pred=y_pred_test,
            y_true=y_test.to_numpy(),
            n_predictors=n_predictors,
            name=self._name,
        )

    @ensure_arg_list_uniqueness()
    def step(
        self,
        direction: Literal["both", "backward", "forward"] = "backward",
        criteria: Literal["aic", "bic"] = "aic",
        kept_vars: list[str] | None = None,
        all_vars: list[str] | None = None,
        start_vars: list[str] | None = None,
        max_steps: int = 100,
    ) -> list[str]:
        """Finish writing description

        Categorical variables will either be included or excluded as a whole.

        Parameters
        ----------
        direction : Literal["both", "backward", "forward"]
            The direction of the stepwise selection. Default: 'backward'.
        criteria : Literal["aic", "bic"]
            The criteria to use for selecting the best model. Default: 'aic'.
        kept_vars : list[str]
            The variables that should be kept in the model. Default: None.
            If None, defaults to empty list.
        all_vars : list[str]
            The variables that are candidates for inclusion in the model. Default: None.
            If None, defaults to all variables in the training data.
        start_vars : list[str]
            The variables to start the bidirectional stepwise selection with.
            Ignored if direction is not 'both'. If direction is 'both' and
            start_vars is None, then the starting variables are the kept_vars.
            Default: None.
        max_steps : int
            The maximum number of steps to take. Default: 100.

        Returns
        -------
        list[str]
            The subset of predictors that are most likely to be significant.
        """
        if max_steps <= 0:
            raise ValueError("max_steps cannot be non-positive")

        X_train, y_train = self._dataemitter.emit_train_Xy()

        # set upper to all possible variables if nothing is specified
        if all_vars is None:
            all_vars = X_train.columns.tolist()
        if kept_vars is None:
            kept_vars = []

        # ensure that kept vars are in all vars
        for var in kept_vars:
            if var not in all_vars:
                raise ValueError(f"{var} is not in all_vars")

        # set our current variables to our starting list
        if direction == "forward":
            included_vars = kept_vars.copy()
        elif direction == "backward":
            included_vars = all_vars.copy()
        elif direction == "both":
            if start_vars is None:
                included_vars = kept_vars.copy()
            else:
                included_vars = start_vars.copy()

        # set our starting score and best models
        current_score = score_ols_model(
            X_train,
            y_train,
            included_vars,
            metric=criteria,
        )
        current_step = 0

        while current_step < max_steps:
            # Forward step
            if direction == "forward":
                excluded = list(set(all_vars) - set(included_vars))

                best_score = current_score
                var_to_add = None
                for new_var in excluded:
                    candidate_features = included_vars + [new_var]
                    score = score_ols_model(
                        X_train,
                        y_train,
                        candidate_features,
                        metric=criteria,
                    )
                    if score < best_score:
                        best_score = score
                        var_to_add = new_var

                # If we didn't find a variable to add (score is not better), break
                if var_to_add is None:
                    break

                included_vars.append(var_to_add)

            # Backward step
            elif direction == "backward":
                if len(included_vars) <= len(kept_vars):
                    break

                best_score = current_score
                var_to_remove = None

                for candidate in included_vars:
                    if candidate in kept_vars:
                        continue

                    candidate_features = included_vars.copy()
                    candidate_features.remove(candidate)
                    score = score_ols_model(
                        X_train,
                        y_train,
                        candidate_features,
                        metric=criteria,
                    )
                    if score < best_score:
                        best_score = score
                        var_to_remove = candidate

                if var_to_remove is None:
                    break

                included_vars.remove(var_to_remove)

            elif direction == "both":
                excluded = list(set(all_vars) - set(included_vars))

                best_score = current_score

                best_forward_score = current_score
                var_to_add = None
                for new_var in excluded:
                    candidate_features = included_vars + [new_var]
                    score = score_ols_model(
                        X_train,
                        y_train,
                        candidate_features,
                        metric=criteria,
                    )
                    if score < best_forward_score:
                        best_forward_score = score
                        var_to_add = new_var

                best_backward_score = current_score
                var_to_remove = None

                for candidate in included_vars:
                    if candidate in kept_vars:
                        continue

                    candidate_features = included_vars.copy()
                    candidate_features.remove(candidate)
                    score = score_ols_model(
                        X_train,
                        y_train,
                        candidate_features,
                        metric=criteria,
                    )
                    if score < best_backward_score:
                        best_backward_score = score
                        var_to_remove = candidate

                if best_forward_score < best_backward_score:
                    if var_to_add is None:
                        break
                    included_vars.append(var_to_add)
                    best_score = best_forward_score
                else:
                    if var_to_remove is None:
                        break
                    included_vars.remove(var_to_remove)
                    best_score = best_backward_score

            current_score = best_score
            current_step += 1

        return included_vars

    def __str__(self):
        return self._name
