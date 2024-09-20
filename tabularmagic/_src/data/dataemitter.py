import copy
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.utils._testing import ignore_warnings
import numpy as np
from typing import Literal
from .preprocessing import (
    BaseSingleVarScaler,
    Log1PTransformSingleVar,
    LogTransformSingleVar,
    MinMaxSingleVar,
    StandardizeSingleVar,
    CustomOneHotEncoder,
)

from ..utils import ensure_arg_list_uniqueness
from ..display.print_utils import (
    print_wrapped,
)


class PreprocessStepTracer:
    """PreprocessStepTracer is a class that tracks preprocessing steps.

    This class is used by the DataHandler class to track all preprocessing
    steps applied to the data, as well as by the DataEmitter class to re-trace
    the steps when emitting data for model fitting.
    """

    def __init__(self):
        """Initializes a PreprocessStepTracer object."""
        self.reset()

    def reset(self):
        """Clears all preprocessing steps."""
        self._steps = []
        self._category_mapping = {}

    def add_step(self, step: str, kwargs: dict):
        """Adds a preprocessing step to the tracer.

        Parameters
        ----------
        step : str
            Preprocessing method name.

        kwargs : dict
            Keyword arguments for the preprocessing method.
        """
        self._steps.append({"step": step, "kwargs": kwargs})

    def add_category_mapping(self, mapping: dict):
        """Adds a category mapping to the tracer.

        Parameters
        ----------
        mapping : dict
            Dictionary with categorical variables as keys and
            categories as values.
        """
        self._category_mapping = mapping.copy()

    def copy(self) -> "PreprocessStepTracer":
        """Returns a copy of the PreprocessStepTracer object.

        Returns
        -------
        PreprocessStepTracer
            Copy of the PreprocessStepTracer object.
        """
        new = PreprocessStepTracer()
        new._steps = self._steps.copy()
        return new


class DataEmitter:
    """DataEmitter is a class that emits data for model fitting and other computational
    methods. By emit, we mean that preprocessing steps are fitted on the training data
    and then applied to the test data. The
    DataEmitter is outputted by DataHandler methods.
    """

    @ensure_arg_list_uniqueness()
    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        y_var: str,
        X_vars: list[str],
        step_tracer: PreprocessStepTracer,
    ):
        """Initializes a DataEmitter object.

        Parameters
        ----------
        df_train : pd.DataFrame
            df_train is the train DataFrame before preprocessing but
            after variable manipulation. DataEmitter copies this DataFrame.

        df_test : pd.DataFrame
            df_test is the train DataFrame before preprocessing but
            after variable manipulation. DataEmitter copies this DataFrame.

        y_var : str
            The target variable.

        X_vars : list[str]
            The predictor variables (before one-hot encoding, if applicable).

        step_tracer: PreprocessStepTracer
        """
        self._working_df_train = df_train.copy()
        self._working_df_test = df_test.copy()

        self._yvar = y_var
        self._Xvars = X_vars

        self._step_tracer = step_tracer

        self._numeric_imputer = None
        self._categorical_imputer = None
        self._highly_missing_vars_dropped = None
        self._onehot_encoder = None
        self._second_onehot_encoder = None
        self._var_to_scaler = dict()

        self._yscaler = None

        (
            self._categorical_vars,
            self._numeric_vars,
            self._categorical_to_categories,
        ) = self._compute_categorical_numeric_vars(self._working_df_train)

        self._pre_onehot_X_vars_subset = None
        self._final_X_vars_subset = None

        self._forward()

    def _forward(self):
        """Applies all preprocessing steps in the step tracer."""
        for step in self._step_tracer._steps:
            if step["step"] == "onehot":
                self._onehot(**step["kwargs"])
            elif step["step"] == "impute":
                self._impute(**step["kwargs"])
            elif step["step"] == "scale":
                self._scale(**step["kwargs"])
            elif step["step"] == "drop_highly_missing_vars":
                self._drop_highly_missing_vars(**step["kwargs"])
            elif step["step"] == "dropna":
                self._dropna(**step["kwargs"])
            elif step["step"] == "force_numeric":
                self._force_numeric(**step["kwargs"])
            elif step["step"] == "force_binary":
                self._force_binary(**step["kwargs"])
            elif step["step"] == "force_categorical":
                self._force_categorical(**step["kwargs"])
            elif step["step"] == "select_vars":
                self._select_vars(**step["kwargs"])
            elif step["step"] == "drop_vars":
                self._drop_vars(**step["kwargs"])
            elif step["step"] == "add_scaler":
                self._add_scaler(**step["kwargs"])
            else:
                raise ValueError("Invalid step.")

    def y_scaler(self) -> BaseSingleVarScaler | None:
        """Returns the scaler for the y variable, which could be None.

        Returns
        -------
        BaseSingleVarScaler | None
            If the y variable is not scaled, returns None.
        """
        return self._yscaler

    def emit_train_test_Xy(
        self,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Returns a tuple as follows:
        (X_train_df, y_train_series, X_test_df, y_test_series).

        ** WARNING **
        THIS METHOD SHOULD BE USED EXCLUSIVELY FOR MODEL FITTING, NOT
        FOR DATA ANALYSIS OR EXPLORATION.

        Cross validation should treat all training data-dependent preprocessing
        methods as part of the model fitting process. This method and class
        is intended to help satisfy that requirement by allowing the
        DataHandler to produce several DataEmitter objects, each of which
        preprocesses the data in the same way, but independently of each other.

        Rows with missing values for any of the X and y variables are dropped.

        If categorical variables are detected in the X DataFrames,
        they will be one-hot encoded.

        Returns
        -------
        pd.DataFrame
            X_train_df: The training DataFrame of predictors.

        pd.Series
            y_train_series: The training Series of the target variable.

        pd.DataFrame
            X_test_df: The test DataFrame of predictors.

        pd.Series
            y_test_series: The test Series of the target variable.
        """
        all_vars = self._Xvars + [self._yvar]
        prev_train_len = len(self._working_df_train)
        working_df_train = self._working_df_train[all_vars].dropna()
        new_train_len = len(working_df_train)
        prev_test_len = len(self._working_df_test)
        working_df_test = self._working_df_test[all_vars].dropna()
        new_test_len = len(working_df_test)
        if prev_train_len != new_train_len:
            print_wrapped(
                f"Train data: dropped {prev_train_len - new_train_len} rows "
                "with missing values "
                f"out of a total of {prev_train_len} rows.",
                type="WARNING",
            )
        if prev_test_len != new_test_len:
            print_wrapped(
                f"Test data: dropped {prev_test_len - new_test_len} rows "
                "with missing values "
                f"out of a total of {prev_test_len} rows.",
                type="WARNING",
            )
        if self._pre_onehot_X_vars_subset is not None:
            xvars = self._pre_onehot_X_vars_subset
        else:
            xvars = self._Xvars
        X_train_df = self._onehot_helper(
            working_df_train[xvars], fit=True, use_second_encoder=True
        )
        X_test_df = self._onehot_helper(
            working_df_test[xvars], fit=False, use_second_encoder=True
        )
        if self._final_X_vars_subset is not None:
            X_train_df = X_train_df[self._final_X_vars_subset]
            X_test_df = X_test_df[self._final_X_vars_subset]
        return (
            X_train_df,
            working_df_train[self._yvar],
            X_test_df,
            working_df_test[self._yvar],
        )

    def emit_train_Xy(self) -> tuple[pd.DataFrame, pd.Series]:
        """Returns a tuple as follows: (X_train_df, y_train_series).

        ** WARNING **
        THIS METHOD SHOULD BE USED EXCLUSIVELY FOR MODEL FITTING, NOT
        FOR DATA ANALYSIS OR EXPLORATION.

        Cross validation should treat all training data-dependent preprocessing
        methods as part of the model fitting process. This method and class
        is intended to help satisfy that requirement by allowing the
        DataHandler to produce several DataEmitter objects, each of which
        preprocesses the data in the same way, but independently of each other.

        Rows with missing values for any of the X and y variables are dropped.

        If categorical variables are detected in the X DataFrames,
        they will be one-hot encoded.

        Returns
        -------
        pd.DataFrame
            X_train_df: The training DataFrame of predictors.

        pd.Series
            y_train_series: The training Series of the target variable.
        """
        all_vars = self._Xvars + [self._yvar]
        prev_train_len = len(self._working_df_train)
        working_df_train = self._working_df_train[all_vars].dropna()
        new_train_len = len(working_df_train)
        if prev_train_len != new_train_len:
            print_wrapped(
                f"Train data: dropped {prev_train_len - new_train_len} rows "
                "with missing values "
                f"out of a total of {prev_train_len} rows.",
                type="WARNING",
            )
        if self._pre_onehot_X_vars_subset is not None:
            xvars = self._pre_onehot_X_vars_subset
        else:
            xvars = self._Xvars
        X_train_df = self._onehot_helper(
            working_df_train[xvars], fit=True, use_second_encoder=True
        )
        if self._final_X_vars_subset is not None:
            X_train_df = X_train_df[self._final_X_vars_subset]
        return X_train_df, working_df_train[self._yvar]

    def emit_test_Xy(self) -> tuple[pd.DataFrame, pd.Series]:
        """Returns a tuple as follows: (X_test_df, y_test_series).

        ** WARNING **
        THIS METHOD SHOULD BE USED EXCLUSIVELY FOR MODEL FITTING, NOT
        FOR DATA ANALYSIS OR EXPLORATION.

        Cross validation should treat all training data-dependent preprocessing
        methods as part of the model fitting process. This method and class
        is intended to help satisfy that requirement by allowing the
        DataHandler to produce several DataEmitter objects, each of which
        preprocesses the data in the same way, but independently of each other.

        Rows with missing values for any of the X and y variables are dropped.

        If categorical variables are detected in the X DataFrames,
        they will be one-hot encoded.

        Returns
        -------
        pd.DataFrame
            X_test_df: The test DataFrame of predictors.

        pd.Series
            y_test_series: The test Series of the target variable.
        """
        all_vars = self._Xvars + [self._yvar]
        prev_test_len = len(self._working_df_test)
        working_df_test = self._working_df_test[all_vars].dropna()
        new_test_len = len(working_df_test)
        if prev_test_len != new_test_len:
            print_wrapped(
                f"Test data: dropped {prev_test_len - new_test_len} rows "
                f"with missing values out of a total of {prev_test_len} rows.",
                type="WARNING",
            )
        if self._pre_onehot_X_vars_subset is not None:
            xvars = self._pre_onehot_X_vars_subset
        else:
            xvars = self._Xvars
        X_test_df = self._onehot_helper(
            working_df_test[xvars], fit=False, use_second_encoder=True
        )
        if self._final_X_vars_subset is not None:
            X_test_df = X_test_df[self._final_X_vars_subset]
        return X_test_df, working_df_test[self._yvar]
    

    def select_predictors_pre_onehot(self, predictors: list[str] | None):
        """Selects a subset of predictors lazily (before one-hot encoding).

        Parameters
        ----------
        predictors : list[str] | None
            List of predictors to select. If None, all predictors are selected.
        """
        self._pre_onehot_X_vars_subset = predictors


    @ensure_arg_list_uniqueness()
    def select_predictors(self, predictors: list[str] | None):
        """Selects a subset of predictors lazily (last step of the emit methods).
        These predictor values should be the POST one-hot encoded values, if applicable.

        Parameters
        ----------
        predictors : list[str] | None
            List of predictors to select. If None, all predictors are selected.
        """
        self._final_X_vars_subset = predictors

    def _onehot(
        self,
        vars: list[str] | None = None,
        dropfirst: bool = True,
        X: pd.DataFrame | None = None,
    ) -> pd.DataFrame | None:
        """One-hot encodes all categorical variables in-place.

        Parameters
        ----------
        vars : list[str]
            Default: None.
            If not None, only one-hot encodes the specified variables.

        dropfirst : bool
            Default: True.
            If True, the first dummy variable is dropped.

        X : pd.DataFrame | None
            Default: None. If not None, one-hot encodes the specified variables
            in X.

        Returns
        -------
        pd.DataFrame | None
            If X is None, returns None. Otherwise, returns the transformed DataFrame.
        """
        if vars is None:
            vars = self._categorical_vars

        if X is not None:
            return self._onehot_helper(X, vars=vars, dropfirst=dropfirst, fit=False)

        self._working_df_train = self._onehot_helper(
            self._working_df_train, vars=vars, dropfirst=dropfirst, fit=True
        )
        self._working_df_test = self._onehot_helper(
            self._working_df_test, vars=vars, dropfirst=dropfirst, fit=False
        )
        (
            self._categorical_vars,
            self._numeric_vars,
            self._categorical_to_categories,
        ) = self._compute_categorical_numeric_vars(self._working_df_train)
        return None

    def _drop_highly_missing_vars(
        self,
        include_vars: list[str] | None = None,
        exclude_vars: list[str] | None = None,
        threshold: float = 0.5,
        X: pd.DataFrame | None = None,
    ) -> pd.DataFrame | None:
        """Drops columns with more than 50% missing values (on train) in-place.

        Parameters
        ----------
        include_vars : list[str] | None
            Default: None. If not None, only drops columns with more than 50% missing
            values in the specified variables. Otherwise, drops columns with more than
            50% missing values in all variables.

        exclude_vars : list[str] | None
            Default: None. If not None, excludes the specified variables from the
            list of variables to drop (which is set to all variables by default).

        threshold : float
            Default: 0.5. Proportion of missing values above which a column is dropped.
            For example, if threshold = 0.2, then columns with more than 20% missing
            values are dropped.

        X : pd.DataFrame | None
            Default: None. If not None, drops columns with more than 50% missing
            values in X.

        Returns
        -------
        pd.DataFrame | None
            If X is None, returns None. Otherwise, returns the transformed DataFrame.
        """
        if X is not None:
            prev_vars = X.columns.to_list()
            kept_vars = list(set(prev_vars) - self._highly_missing_vars_dropped)
            return X[kept_vars]

        if include_vars is None:
            prev_vars = self._working_df_train.columns.to_list()
        else:
            prev_vars = include_vars

        if exclude_vars is not None:
            prev_vars = list(set(prev_vars) - set(exclude_vars))

        missingness = self._working_df_train[prev_vars].isna().mean()
        vars_to_drop = missingness[missingness >= threshold].index.to_list()

        if self._highly_missing_vars_dropped is None:
            self._highly_missing_vars_dropped = vars_to_drop
        else:
            self._highly_missing_vars_dropped = self._highly_missing_vars_dropped.add(
                vars_to_drop
            )

        self._working_df_train = self._working_df_train.drop(vars_to_drop, axis=1)
        self._working_df_test = self._working_df_test.drop(vars_to_drop, axis=1)
        (
            self._categorical_vars,
            self._numeric_vars,
            self._categorical_to_categories,
        ) = self._compute_categorical_numeric_vars(self._working_df_train)
        return self

    def _dropna(
        self, vars: list[str], X: pd.DataFrame | None = None
    ) -> pd.DataFrame | None:
        """Drops rows with missing values in-place.

        Parameters
        ----------
        vars : list[str]
            List of variables along which to drop rows with missing values.

        X : pd.DataFrame | None
            Default: None. If not None, drops rows with missing values in X.

        Returns
        -------
        pd.DataFrame | None
            If X is None, returns None. Otherwise, returns the imputed DataFrame.
        """
        if X is not None:
            return X.dropna(subset=vars)

        self._working_df_train = self._working_df_train.dropna(subset=vars)
        self._working_df_test = self._working_df_test.dropna(subset=vars)
        return None

    def _impute(
        self,
        vars: list[str],
        numeric_strategy: Literal["median", "mean", "5nn"] = "median",
        categorical_strategy: Literal["most_frequent"] = "most_frequent",
        X: pd.DataFrame | None = None,
    ) -> pd.DataFrame | None:
        """Imputes missing values in-place.

        Parameters
        ----------
        vars : list[str]
            List of variables to impute missing values.

        numeric_strategy : Literal['median', 'mean', '5nn']
            Default: 'median'.
            Strategy for imputing missing values in numeric variables.
            - 'median': impute with median.
            - 'mean': impute with mean.
            - '5nn': impute with 5-nearest neighbors.

        categorical_strategy : Literal['most_frequent'].
            Default: 'most_frequent'.
            Strategy for imputing missing values in categorical variables.

        X : pd.DataFrame | None
            Default: None. If not None, imputes missing values in X. In this case,
            the imputers must have already been fitted on the training data.

        Returns
        -------
        pd.DataFrame | None
            If X is None, returns None. Otherwise, returns the imputed DataFrame.
        """
        numeric_vars = self._numeric_vars
        categorical_vars = self._categorical_vars
        var_set = set(vars)
        numeric_vars = list(var_set & set(numeric_vars))
        categorical_vars = list(var_set & set(categorical_vars))

        if X is not None:
            if len(numeric_vars) > 0:
                X[numeric_vars] = self._numeric_imputer.transform(X[numeric_vars])
            if len(categorical_vars) > 0:
                X[categorical_vars] = self._categorical_imputer.transform(
                    X[categorical_vars]
                )
            return X

        # impute numeric variables
        if len(numeric_vars) > 0:
            if numeric_strategy == "5nn":
                imputer = KNNImputer(n_neighbors=5, keep_empty_features=True)
            else:
                imputer = SimpleImputer(
                    strategy=numeric_strategy, keep_empty_features=True
                )
            self._working_df_train[numeric_vars] = imputer.fit_transform(
                self._working_df_train[numeric_vars]
            )
            self._working_df_test[numeric_vars] = imputer.transform(
                self._working_df_test[numeric_vars]
            )

        self._numeric_imputer = imputer

        # impute categorical variables
        if len(categorical_vars) > 0:
            imputer = SimpleImputer(
                strategy=categorical_strategy, keep_empty_features=True
            )
            self._working_df_train[categorical_vars] = imputer.fit_transform(
                self._working_df_train[categorical_vars]
            )
            self._working_df_test[categorical_vars] = imputer.transform(
                self._working_df_test[categorical_vars]
            )

        self._categorical_imputer = imputer

        return None

    def _scale(
        self,
        vars: list[str],
        strategy: Literal["standardize", "minmax", "log", "log1p"] = "standardize",
        X: pd.DataFrame | None = None,
    ) -> pd.DataFrame | None:
        """Scales variable values.

        Parameters
        ----------
        vars : list[str]
            List of variables to scale. If None, scales all numeric
            variables.

        strategy : Literal['standardize', 'minmax', 'log', 'log1p']

        X : pd.DataFrame | None
            Default: None. If not None, scales the specified variables in X.

        Returns
        -------
        pd.DataFrame | None
            If X is None, returns None. Otherwise, returns the scaled DataFrame.
        """
        for var in vars:
            if var not in self._numeric_vars:
                print_wrapped(
                    f"Variable {var} is not numeric. Skipping.", type="WARNING"
                )
                continue

            if X is not None:
                if var in self._var_to_scaler:
                    X[var] = self._var_to_scaler[var].transform(X[var].to_numpy())
                continue

            train_data = self._working_df_train[var].to_numpy()
            if strategy == "standardize":
                scaler = StandardizeSingleVar(var, train_data)
            elif strategy == "minmax":
                scaler = MinMaxSingleVar(var, train_data)
            elif strategy == "log":
                scaler = LogTransformSingleVar(var, train_data)
            elif strategy == "log1p":
                scaler = Log1PTransformSingleVar(var, train_data)
            else:
                raise ValueError("Invalid scaling strategy.")

            self._var_to_scaler[var] = scaler

            self._working_df_train[var] = scaler.transform(
                self._working_df_train[var].to_numpy()
            )
            self._working_df_test[var] = scaler.transform(
                self._working_df_test[var].to_numpy()
            )

            if var == self._yvar:
                self._yscaler = scaler

        return X

    def _select_vars(
        self, vars: list[str], X: pd.DataFrame | None = None
    ) -> pd.DataFrame | None:
        """Selects subset of (column) variables in-place on the working
        train and test DataFrames.

        Parameters
        ----------
        vars : list[str]

        X : pd.DataFrame | None
            Default: None. If not None, selects the specified variables in X.

        Returns
        -------
        pd.DataFrame | None
            If X is None, returns None. Otherwise, returns the transformed DataFrame.
        """
        if X is not None:
            return X[vars]

        self._working_df_test = self._working_df_test[vars]
        self._working_df_train = self._working_df_train[vars]
        (
            self._categorical_vars,
            self._numeric_vars,
            self._categorical_to_categories,
        ) = self._compute_categorical_numeric_vars(self._working_df_train)
        return None

    def _drop_vars(
        self, vars: list[str], X: pd.DataFrame | None = None
    ) -> pd.DataFrame | None:
        """Drops subset of variables (columns) in-place on the working
        train and test DataFrames.

        Parameters
        ----------
        vars : list[str]

        Returns
        -------
        pd.DataFrame | None
            If X is None, returns None. Otherwise, returns the transformed DataFrame.
        """
        if X is not None:
            return X.drop(vars, axis="columns")

        self._working_df_test = self._working_df_test.drop(vars, axis="columns")
        self._working_df_train = self._working_df_train.drop(vars, axis="columns")
        (
            self._categorical_vars,
            self._numeric_vars,
            self._categorical_to_categories,
        ) = self._compute_categorical_numeric_vars(self._working_df_train)
        return None

    def _force_numeric(
        self, vars: list[str], X: pd.DataFrame | None = None
    ) -> pd.DataFrame | None:
        """Forces variables to numeric (floats).

        Parameters
        ----------
        vars : list[str]
            Name of variables.

        X : pd.DataFrame | None
            Default: None. If not None, forces the specified variables to numeric in X.

        Returns
        -------
        pd.DataFrame | None
            If X is None, returns None. Otherwise, returns the transformed DataFrame.
        """
        for var in vars:
            if var not in self._working_df_train.columns:
                raise ValueError(f"Invalid variable name: {var}.")
            try:
                if X is not None:
                    X[var] = X[var].apply(lambda x: float(x) if pd.notna(x) else np.nan)
                    continue
                self._working_df_train[var] = self._working_df_train[var].apply(
                    lambda x: float(x) if pd.notna(x) else np.nan
                )
                self._working_df_test[var] = self._working_df_test[var].apply(
                    lambda x: float(x) if pd.notna(x) else np.nan
                )
            except Exception:
                pass
        return X

    def _force_binary(
        self,
        vars: list[str],
        pos_labels: list[str] | None = None,
        ignore_multiclass: bool = False,
        rename: bool = False,
        X: pd.DataFrame | None = None,
    ) -> pd.DataFrame | None:
        """Forces variables to be binary (0 and 1 valued numeric variables).
        Does nothing if the data contains more than two classes unless
        ignore_multiclass is True and pos_label is specified,
        in which case all classes except pos_label are labeled with zero.

        Parameters
        ----------
        vars : list[str]
            Name of variables to force to binary.

        pos_labels : list[str]
            Default: None. The positive labels.
            If None, the first class for each var is the positive label.

        ignore_multiclass : bool
            Default: False. If True, all classes except pos_label are labeled with
            zero. Otherwise raises ValueError.

        rename : bool
            Default: False. If True, the variables are renamed to
            {pos_label}::{var}.

        X : pd.DataFrame | None
            Default: None. If not None, forces the specified variables to binary

        Returns
        -------
        pd.DataFrame | None
            If X is None, returns None. Otherwise, returns the transformed DataFrame.
        """
        if pos_labels is None and ignore_multiclass:
            raise ValueError(
                "pos_labels must be specified if ignore_multiclass is True."
            )

        vars_to_renamed = {}
        for i, var in enumerate(vars):
            if var not in self._working_df_train.columns:
                raise ValueError(f"Invalid variable name: {var}.")

            if pos_labels is None:
                unique_vals = self._working_df_train[var].unique()
                if len(unique_vals) > 2:
                    continue
                pos_label = unique_vals[0]

                if X is not None:
                    X[var] = X[var].apply(lambda x: 1 if x == pos_label else 0)
                    continue

                self._working_df_train[var] = self._working_df_train[var].apply(
                    lambda x: 1 if x == pos_label else 0
                )
                self._working_df_test[var] = self._working_df_test[var].apply(
                    lambda x: 1 if x == pos_label else 0
                )

            else:
                unique_vals = self._working_df_train[var].unique()
                if len(unique_vals) > 2:
                    if not ignore_multiclass:
                        continue
                pos_label = pos_labels[i]

                if X is not None:
                    X[var] = X[var].apply(lambda x: 1 if x == pos_label else 0)
                    continue

                self._working_df_train[var] = self._working_df_train[var].apply(
                    lambda x: 1 if x == pos_label else 0
                )
                self._working_df_test[var] = self._working_df_test[var].apply(
                    lambda x: 1 if x == pos_label else 0
                )

            vars_to_renamed[var] = f"{pos_label}::{var}"

        if X is not None:
            return X

        if rename:
            self._working_df_train = self._working_df_train.rename(
                columns=vars_to_renamed
            )
            self._working_df_test = self._working_df_test.rename(
                columns=vars_to_renamed
            )

        (
            self._categorical_vars,
            self._numeric_vars,
            self._categorical_to_categories,
        ) = self._compute_categorical_numeric_vars(self._working_df_train)
        return None

    def _force_categorical(
        self, vars: list[str], X: pd.DataFrame | None = None
    ) -> pd.DataFrame | None:
        """Forces variables to become categorical.
        Example use case: create numericly-coded categorical variables.

        Parameters
        ----------
        vars : list[str]
            Name of variables.

        X : pd.DataFrame | None
            Default: None. If not None, forces the specified variables to categorical
            in X.

        Returns
        -------
        pd.DataFrame | None
            If X is None, returns None. Otherwise, returns the transformed DataFrame.
        """
        if not isinstance(vars, list):
            vars = [vars]
        for var in vars:
            if X is not None:
                X[var] = X[var].apply(lambda x: str(x) if pd.notna(x) else np.nan)
                continue
            self._working_df_train[var] = self._working_df_train[var].apply(
                lambda x: str(x) if pd.notna(x) else np.nan
            )
            self._working_df_test[var] = self._working_df_test[var].apply(
                lambda x: str(x) if pd.notna(x) else np.nan
            )

        if X is not None:
            return X

        (
            self._categorical_vars,
            self._numeric_vars,
            self._categorical_to_categories,
        ) = self._compute_categorical_numeric_vars(self._working_df_train)

        return None

    def _compute_categories(
        self, df: pd.DataFrame, categorical_vars: list[str]
    ) -> dict:
        """Returns a dictionary containing the categorical variables
        each mapped to a list of all categories in the variable.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame.

        categorical_vars : list[str]
            List of categorical variable names.

        Returns
        -------
        dict
            Dictionary with categorical variables as keys and
            categories as values.
        """
        categories_dict = {}
        for var in categorical_vars:
            categories_dict[var] = df[var].unique().tolist()
        return categories_dict

    def _onehot_helper(
        self,
        df: pd.DataFrame,
        vars: list[str] | None = None,
        dropfirst: bool = True,
        fit: bool = True,
        use_second_encoder: bool = False,
    ) -> pd.DataFrame:
        """One-hot encodes all categorical variables with more than
        two categories.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame.

        vars : list[str]
            Default: None. If not None, only one-hot encodes the specified variables.

        dropfirst : bool
            Default: True. If True, the first dummy variable is dropped.

        fit : bool
            Default: True.
            If True, fits the encoder on the training data. Otherwise,
            only transforms the test data.

        use_second_encoder : bool
            Default: False. If True, uses a second encoder. The second encoder
            is useful for emitting data that must be one-hot encoded but was
            not one-hot encoded by the user.

        Returns
        -------
        pd.DataFrame
            The DataFrame with one-hot encoded variables.
        """
        if vars is None:
            categorical_vars = df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.to_list()
        else:
            for var in vars:
                if var not in df.columns:
                    raise ValueError(f"Invalid variable name: {var}")
            categorical_vars = vars

        if use_second_encoder:
            encoder = self._second_onehot_encoder
        else:
            encoder = self._onehot_encoder

        if categorical_vars:
            if dropfirst:
                drop = "first"
            else:
                drop = "if_binary"

            if fit:
                encoder = CustomOneHotEncoder(
                    drop=drop, sparse_output=False, handle_unknown="ignore"
                )
                encoded = encoder.fit_transform(df[categorical_vars])
                feature_names = encoder.get_feature_names_out(categorical_vars)
                df_encoded = pd.DataFrame(
                    encoded, columns=feature_names, index=df.index
                )

            else:
                encoded = ignore_warnings(encoder.transform)(df[categorical_vars])
                feature_names = encoder.get_feature_names_out(categorical_vars)
                df_encoded = pd.DataFrame(
                    encoded, columns=feature_names, index=df.index
                )

            if use_second_encoder:
                self._second_onehot_encoder = encoder
            else:
                self._onehot_encoder = encoder

            return pd.concat([df_encoded, df.drop(columns=categorical_vars)], axis=1)
        else:
            return df

    def _compute_categorical_numeric_vars(
        self, df: pd.DataFrame
    ) -> tuple[list[str], list[str], dict]:
        """Returns the categorical and numeric columns.
        Also returns the categorical variables mapped to their categories.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame.

        Returns
        -------
        list[str]
            categorical_vars : List of categorical variables.

        list[str]
            numeric_vars : List of numeric variables.

        dict
            categorical_mapped : Dictionary with categorical variables as keys
        """
        categorical_vars = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.to_list()
        numeric_vars = df.select_dtypes(
            exclude=["object", "category", "bool"]
        ).columns.to_list()
        categorical_mapped = self._compute_categories(df, categorical_vars)
        return categorical_vars, numeric_vars, categorical_mapped

    def _add_scaler(self, scaler: BaseSingleVarScaler, var: str) -> "DataEmitter":
        """Adds a scaler for the target variable.

        Parameters
        ----------
        scaler : BaseSingleVarScaler
            Scaler object.

        var : str
            Name of the variable.

        Returns
        -------
        DataEmitter
            Returns self for method chaining.
        """
        if var == self._yvar:
            self._yscaler = scaler
        return self
    
    def X_vars(self) -> list[str]:
        """Returns the predictor variables.

        Returns
        -------
        list[str]
            The predictor variables.
        """
        return self._Xvars
    
    def y_var(self) -> str:
        """Returns the target variable.

        Returns
        -------
        str
            The target variable.
        """
        return self._yvar
    
    def vars(self) -> list[str]:
        """Returns all variables.

        Returns
        -------
        list[str]
            All variables.
        """
        return self._Xvars + [self._yvar]

    def custom_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms the input DataFrame using the preprocessing steps
        in the step tracer.

        Parameters
        ----------
        X : pd.DataFrame
            The DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame.
        """

        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a DataFrame.")

        X = X.copy()

        for step in self._step_tracer._steps:
            if step["step"] == "onehot":
                X = self._onehot(X=X, **step["kwargs"])
            elif step["step"] == "impute":
                X = self._impute(X=X, **step["kwargs"])
            elif step["step"] == "scale":
                X = self._scale(X=X, **step["kwargs"])
            elif step["step"] == "drop_highly_missing_vars":
                X = self._drop_highly_missing_vars(X=X, **step["kwargs"])
            elif step["step"] == "force_numeric":
                X = self._force_numeric(X=X, **step["kwargs"])
            elif step["step"] == "force_binary":
                X = self._force_binary(X=X, **step["kwargs"])
            elif step["step"] == "force_categorical":
                X = self._force_categorical(X=X, **step["kwargs"])
            elif step["step"] == "select_vars":
                X = self._select_vars(X=X, **step["kwargs"])
            elif step["step"] == "drop_vars":
                X = self._drop_vars(X=X, **step["kwargs"])
            elif step["step"] == "add_scaler":
                X = self._add_scaler(X=X, **step["kwargs"])
            else:
                raise ValueError("Invalid step.")

        X = X[self._Xvars]

        X = X.dropna()
        X = self._onehot_helper(X[self._Xvars], fit=False, use_second_encoder=True)

        if self._final_X_vars_subset is not None:
            X = X[self._final_X_vars_subset]

        return X

    def sklearn_preprocessing_transformer(self) -> FunctionTransformer:
        """Builds a FunctionTransformer object for preprocessing.

        Returns
        -------
        FunctionTransformer
            FunctionTransformer object.
        """
        new_emitter = self.copy()
        del new_emitter._working_df_train
        del new_emitter._working_df_test

        custom_transformer = FunctionTransformer(
            new_emitter.custom_transform, validate=False, check_inverse=False
        )

        return custom_transformer

    def copy(self) -> "DataEmitter":
        """Returns a copy of the DataEmitter object.

        Returns
        -------
        DataEmitter
            The copied DataEmitter object.
        """
        return copy.deepcopy(self)
