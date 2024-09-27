import statsmodels.api as sm
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from ..metrics.classification_scoring import ClassificationBinaryScorer
from ..data.datahandler import DataEmitter
import pandas as pd
from typing import Literal
from ..utils import ensure_arg_list_uniqueness, is_numerical
from .lmutils.score import score_model


class LogitLinearModel:
    """Statsmodels Logit wrapper."""

    def __init__(self, name: str | None = None):
        """
        Initializes a LogitLinearModel object. Regresses y on X.

        Parameters
        ----------
        name : str
            Default: None. Determines how the model shows up in the reports.
            If None, the name is set to be the class name.
        """
        self.estimator = None
        self._name = name
        self._label_encoder = None
        if self._name is None:
            self._name = "Logit Linear Model"

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

        # Emit all data
        X_train, y_train = self._dataemitter.emit_train_Xy()
        X_test, y_test = self._dataemitter.emit_test_Xy()

        # we force the constant to be included
        X_train = sm.add_constant(X_train, has_constant="add")
        X_test = sm.add_constant(X_test, has_constant="add")

        y_levels = y_train.unique()
        if y_levels.size != 2:
            raise ValueError("Target variable must have 2 levels")

        y_test_levels = y_test.unique()
        if y_test_levels.size > 2:
            raise ValueError(
                "Target variable in test set detected to have more than 2 levels"
            )

        # we allow y_train to be categorical, i.e. we encode it with a label encoder
        self._y_label_order = None
        if not is_numerical(y_train):
            self._label_encoder = LabelEncoder()
            y_train = self._label_encoder.fit_transform(y_train)
            self._y_label_order = self._label_encoder.classes_
            y_test = self._label_encoder.transform(y_test)

        self.estimator = sm.Logit(y_train, X_train).fit(cov_type="HC3")

        y_pred_train: np.ndarray = self.estimator.predict(exog=X_train).to_numpy()

        # Find the best threshold based on f1 score
        best_score = None
        best_threshold = None
        for temp_threshold in np.linspace(0.0, 1.0, num=21):
            y_pred_train_binary = (y_pred_train > temp_threshold).astype(int)
            curr_score = f1_score(y_train, y_pred_train_binary)
            if best_score is None or curr_score > best_score:
                best_score = curr_score
                best_threshold = temp_threshold

        y_pred_train_binary = (y_pred_train >= best_threshold).astype(int)

        self.train_scorer = ClassificationBinaryScorer(
            y_pred=y_pred_train_binary,
            y_true=y_train,
            pos_label=self._y_label_order[1] if self._y_label_order is not None else 1,
            y_pred_score=np.hstack(
                [
                    np.zeros(shape=(len(y_pred_train), 1)),
                    y_pred_train.reshape(-1, 1),
                ]
            ),
            name=self._name,
        )

        y_pred_test = self.estimator.predict(X_test).to_numpy()
        y_pred_test_binary = (y_pred_test >= best_threshold).astype(int)

        self.test_scorer = ClassificationBinaryScorer(
            y_pred=y_pred_test_binary,
            y_true=y_test,
            pos_label=self._y_label_order[1] if self._y_label_order is not None else 1,
            y_pred_score=np.hstack(
                [np.zeros(shape=(len(y_pred_test), 1)), y_pred_test.reshape(-1, 1)]
            ),
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
        """This method implements stepwise selection for identifying important
        features. If the direction is set to forward, the algorithm will start
        with no selected variables and will at each time step add every
        left-out feature to the model separately. the left-out feature
        that results in the best improvement in the metric (aic or bic) will
        be selected as an important feature. This happens until all variables
        are added or adding a left-out variable does not improve the metric
        of choice.

        If the direction is set to backward, the algorithm will start with all
        variables selected and will at each time step remove each included
        variable separately. The variable that results in the best improvement
        in the metric when removed from the model will be removed from the
        list of selected features.

        Categorical variables will either be included or excluded as a whole.

        Parameters
        ----------
        direction : Literal["both", "backward", "forward"]
            Default: 'backward'. The direction of the stepwise selection.

        criteria : Literal["aic", "bic"]
            Default: 'aic'. The criteria to use for selecting the best model.

        kept_vars : list[str]
            Default: None. The variables that should be kept in the model.
            If None, defaults to empty list.

        all_vars : list[str]
            Default: None. The variables that are candidates for inclusion in the model.
            If None, defaults to all variables in the training data.

        start_vars : list[str]
            Default: None. The variables to start the bidirectional stepwise selection with.
            Ignored if direction is not 'both'. If direction is 'both' and
            start_vars is None, then the starting variables are the kept_vars.

        max_steps : int
            Default: 100. The maximum number of steps to take.

        Returns
        -------
        list[str]
            The subset of predictors that are most likely to be significant.
        """
        if max_steps <= 0:
            raise ValueError("max_steps cannot be non-positive")

        # set upper to all possible variables if nothing is specified
        local_dataemitter = self._dataemitter.copy()
        if all_vars is None:
            all_vars = local_dataemitter.X_vars()
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
        else:
            raise ValueError("direction must be 'both', 'backward', or 'forward'")

        # set our starting score and best models
        current_score = score_model(
            local_dataemitter,
            included_vars,
            model="logit",
            metric=criteria,
            y_label_encoder=self._label_encoder,
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
                    score = score_model(
                        local_dataemitter,
                        candidate_features,
                        model="logit",
                        metric=criteria,
                        y_label_encoder=self._label_encoder,
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
                    score = score_model(
                        local_dataemitter,
                        candidate_features,
                        model="logit",
                        metric=criteria,
                        y_label_encoder=self._label_encoder,
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
                    score = score_model(
                        local_dataemitter,
                        candidate_features,
                        model="logit",
                        metric=criteria,
                        y_label_encoder=self._label_encoder,
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
                    score = score_model(
                        local_dataemitter,
                        candidate_features,
                        model="logit",
                        metric=criteria,
                        y_label_encoder=self._label_encoder,
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

    def coefs(
        self,
        format: Literal[
            "coef(se)|pval", "coef|se|pval", "coef(ci)|pval", "coef|ci_low|ci_high|pval"
        ] = "coef(se)|pval",
    ) -> pd.DataFrame:
        """Returns a DataFrame containing the coefficients, standard errors, 
        and p-values. The standard errors and p-values are heteroskedasticity-
        robust. Confidence intervals are reported at 95% confidence level if 
        applicable.
        
        Parameters
        ----------
        format : Literal["coef(se)|pval", "coef|se|pval", "coef(ci)|pval",\
                            "coef|ci_low|ci_high|pval"]
            Default: "coef(se)|pval". The format of the output DataFrame.
        """
        params = self.estimator.params
        std_err = self.estimator.bse
        p_values = self.estimator.pvalues

        ci_low = params - 1.96 * std_err
        ci_high = params + 1.96 * std_err

        output_df = pd.DataFrame(
            {
                "coef": params,
                "se": std_err,
                "pval": p_values,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

        if format == "coef(se)|pval":
            output_df["coef(se)"] = output_df.apply(
                lambda row: f"{row['coef']} ({row['se']})", axis=1
            )
            output_df = output_df[["coef(se)", "pval"]]
            output_df = output_df.rename(
                columns={"coef(se)": "Coefficient (Std. Error)", "pval": "P-value"}
            )
        elif format == "coef|se|pval":
            output_df = output_df.rename(
                columns={"coef": "Coefficient", "se": "Std. Error", "pval": "P-value"}
            )
        elif format == "coef(ci)|pval":
            output_df["ci_str"] = output_df["coef"] + 1.96 * output_df["se"]
            output_df["coef(ci)"] = output_df.apply(
                lambda row: f"{row['coef']} ({row['ci_low']}, {row['ci_high']})", axis=1
            )
            output_df = output_df[["coef(ci)", "pval"]]
            output_df = output_df.rename(
                columns={"coef(ci)": "Coefficient (95% CI)", "pval": "P-value"}
            )
        elif format == "coef|ci_low|ci_high|pval":
            output_df = output_df.rename(
                columns={
                    "coef": "Coefficient",
                    "ci_low": "CI Lower Bound",
                    "ci_high": "CI Upper Bound",
                    "pval": "P-value",
                }
            )
        else:
            raise ValueError("Invalid format")

        return output_df

    def __str__(self):
        return self._name
