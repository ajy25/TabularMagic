import pandas as pd
import numpy as np
from typing import Literal


from .report import CausalReport
from .causalutils.methods import compute_weights_from_propensity_scores
from .causalutils.models import _SimpleOLS

from ..._src.ml.predict.classification import BaseC, LinearC
from ..._src.data.datahandler import DataHandler


class CausalModel:
    """A class for estimating causal effects."""

    def __init__(
        self,
        datahandler: DataHandler,
        treatment: str,
        outcome: str,
        confounders: list[str],
        dataset: Literal["train", "test", "all"] = "all",
    ):
        """Initializes a CausalModel object.

        Parameters
        ----------
        datahandler: DataHandler
            The DataHandler object. Both the training and test data will be used
            for causal inference (i.e. we will concatenate the training and test
            data).

        treatment : str
            The name of the treatment variable.

        outcome : str
            The name of the outcome variable.

        confounders : list[str]
            The names of the confounding variables.

        dataset : Literal["train", "test", "all"]
            The dataset to use for causal inference. "train" for training data,
            "test" for test data, and "all" for both training and test data,
            by default "all".
        """
        temp_datahandler = datahandler.copy().dropna(
            include_vars=[treatment, outcome] + confounders
        )
        self._treatment = treatment
        self._outcome = outcome
        self._confounders = confounders
        if dataset == "train":
            self._emitter = temp_datahandler.train_test_emitter(
                y_var=outcome,
                X_vars=confounders + [treatment],
            )
            self._propensity_emitter = temp_datahandler.train_test_emitter(
                y_var=treatment,
                X_vars=confounders,
            )
            self._X_df, self._y_df = self._emitter.emit_train_Xy()
        elif dataset == "test":
            self._emitter = temp_datahandler.train_test_emitter(
                y_var=outcome,
                X_vars=confounders + [treatment],
            )
            self._propensity_emitter = temp_datahandler.train_test_emitter(
                y_var=treatment,
                X_vars=confounders,
            )
            self._X_df, self._y_df = self._emitter.emit_test_Xy()
        elif dataset == "all":
            self._emitter = temp_datahandler.full_dataset_emitter(
                y_var=outcome,
                X_vars=confounders + [treatment],
            )
            self._propensity_emitter = temp_datahandler.full_dataset_emitter(
                y_var=treatment,
                X_vars=confounders,
            )
            self._X_df, self._y_df = self._emitter.emit_train_Xy()
        else:
            raise ValueError("Invalid dataset.")

    def estimate_ate(
        self,
        method: Literal["outcome_regression", "ipw_weighted_regression"],
        propensity_score_estimator: BaseC = LinearC(
            type="no_penalty",
        ),
    ) -> CausalReport:
        """Estimates the average treatment effect (ATE).

        Parameters
        ----------
        method : Literal["outcome_regression", "ipw_weighted_regression"]
            The method for estimating the ATE.

        propensity_score_estimator : BaseC, optional
            The estimator/model for computing/predicting the propensity scores,
            by default LinearC(type="no_penalty") (logistic regression).
            Hyperparameters will be selected as specified in the BaseC model.
        """
        if method == "outcome_regression":
            return self._outcome_regression(estimand="ate")
        elif method == "ipw_weighted_regression":
            return self._ipw_weighted_regression(
                estimand="ate", propensity_score_estimator=propensity_score_estimator
            )
        else:
            raise ValueError("Invalid method.")

    def _outcome_regression(
        self,
        estimand: Literal["ate", "att"],
    ) -> CausalReport:
        """Outcome regression with OLS. Confounders are included as predictors.

        Parameters
        ----------
        estimand : Literal["ate", "att"]
            The estimand. "ate" for average treatment effect and "att" for average
            treatment effect on the treated.
        """
        method_description = (
            "Outcome regression with OLS. "
            "Fits a linear regression model to the outcome variable using the "
            "treatment variable and confounders as predictors."
        )

        df_X, df_y = self._X_df, self._y_df

        if estimand == "att":
            df_X = df_X[df_X[self._treatment] == 1]
            df_y = df_y[df_X[self._treatment] == 1]

        model = _SimpleOLS(y=df_y, X=df_X)

        estimate = model.get_coef(self._treatment)
        se = model.get_se(self._treatment)

        return CausalReport(
            estimate=estimate,
            se=se,
            n_units=len(df_X),
            n_units_treated=len(df_X[df_X[self._treatment] == 1]),
            outcome_var=self._outcome,
            treatment_var=self._treatment,
            confounders=self._confounders,
            estimand=estimand,
            method="Outcome Regression",
            method_description=method_description,
        )

    def _ipw_weighted_regression(
        self,
        estimand: Literal["ate", "att"],
        propensity_score_estimator: BaseC = LinearC(
            type="no_penalty",
        ),
    ) -> CausalReport:
        """Inverse probability weighting (IPW)-weighted regression (WLS).

        Parameters
        ----------
        estimand : Literal["ate", "att"]
            The estimand. "ate" for average treatment effect and "att" for average
            treatment effect on the treated.

        propensity_score_estimator : BaseC, optional
            The estimator/model for computing/predicting the propensity scores,
            by default LinearC(type="no_penalty") (logistic regression).
            Hyperparameters will be selected as specified in the model.
        """
        method_description = (
            "Inverse probability weighting (IPW)-weighted regression (WLS). "
            "Weights units via the inverse of their propensity scores. "
            "Uses logistic regression to estimate the propensity scores. "
            f"The logistic regression model is `{str(propensity_score_estimator)}.`"
        )

        emitter = self._emitter

        df_X, df_y = emitter.emit_train_Xy()
        treatment_series = df_X[self._treatment]

        propensity_emitter = self._propensity_emitter
        propensity_score_estimator.specify_data(dataemitter=propensity_emitter)
        propensity_score_estimator.fit()

        propensity_scores_series = pd.Series(
            propensity_score_estimator._train_scorer._y_pred_score, index=df_X.index
        )

        weights = compute_weights_from_propensity_scores(
            propensity_scores=propensity_scores_series,
            treatment=treatment_series,
            estimand=estimand,
        )

        model = _SimpleOLS(y=df_y, X=df_X, weights=weights)

        estimate = model.get_coef(self._treatment)
        se = model.get_se(self._treatment)

        return CausalReport(
            estimate=estimate,
            se=se,
            n_units=len(df_X),
            n_units_treated=len(df_X[df_X[self._treatment] == 1]),
            outcome_var=self._outcome,
            treatment_var=self._treatment,
            confounders=self._confounders,
            estimand=estimand,
            method="Inverse Probability Weighting (IPW)-Weighted Regression",
            method_description=method_description,
        )
