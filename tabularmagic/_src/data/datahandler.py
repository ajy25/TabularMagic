import pandas as pd
import numpy as np
from typing import Literal
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils._testing import ignore_warnings
from ..display.print_utils import (
    print_wrapped,
    color_text,
    bold_text,
    list_to_string,
    fill_ignore_format,
)
from .preprocessing import (
    BaseSingleVarScaler,
    Log1PTransformSingleVar,
    LogTransformSingleVar,
    MinMaxSingleVar,
    StandardizeSingleVar,
    CustomOneHotEncoder,
)
from ..display.print_options import print_options
from ..utils import ensure_arg_list_uniqueness


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
            The predictor variables.

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
        X_train_df = self._onehot_helper(
            working_df_train[self._Xvars], fit=True, use_second_encoder=True
        )
        X_test_df = self._onehot_helper(
            working_df_test[self._Xvars], fit=False, use_second_encoder=True
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
        X_train_df = self._onehot_helper(
            working_df_train[self._Xvars], fit=True, use_second_encoder=True
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
        X_test_df = self._onehot_helper(
            working_df_test[self._Xvars], fit=False, use_second_encoder=True
        )
        if self._final_X_vars_subset is not None:
            X_test_df = X_test_df[self._final_X_vars_subset]
        return X_test_df, working_df_test[self._yvar]

    @ensure_arg_list_uniqueness()
    def select_predictors(self, predictors: list[str] | None):
        """Selects a subset of predictors lazily (last step of the emit methods).

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
        self, threshold: float = 0.5, X: pd.DataFrame | None = None
    ) -> pd.DataFrame | None:
        """Drops columns with more than 50% missing values (on train) in-place.

        Parameters
        ----------
        threshold : float
            Default: 0.5. Proportion of missing values above which a column is dropped.

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
            kept_vars = set(prev_vars) - self._highly_missing_vars_dropped
            return X[kept_vars]

        prev_vars = self._working_df_train.columns.to_list()
        self._working_df_train = self._working_df_train.dropna(
            axis=1, thresh=threshold * len(self._working_df_train)
        )
        curr_vars = self._working_df_train.columns.to_list()
        self._highly_missing_vars_dropped = set(prev_vars) - set(curr_vars)

        self._working_df_test = self._working_df_test.drop(
            self._highly_missing_vars_dropped, axis=1
        )
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
            {pos_label}_yn({var}).

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

            vars_to_renamed[var] = f"{pos_label}_yn({var})"

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

        custom_transformer = FunctionTransformer(
            self.custom_transform, validate=False, check_inverse=False
        )

        return custom_transformer


class DataHandler:
    """DataHandler: a class that handles all aspects of data preprocessing and loading."""

    def __init__(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        name: str | None = None,
        verbose: bool = True,
    ):
        """Initializes a DataHandler object.

        Parameters
        ----------
        df_train : pd.DataFrame
            The train DataFrame.

        df_test : pd.DataFrame
            The test DataFrame.

        name : str | None
            Default: None. The name of the DataHandler object.

        verbose : bool
            Default: True. If True, prints updates and warnings.
        """
        self._checkpoint_name_to_df: dict[
            str, tuple[pd.DataFrame, pd.DataFrame]
        ] = dict()
        self._verbose = verbose

        # verify and set the original DataFrames
        self._verify_input_dfs(df_train, df_test)

        self._orig_df_train, self._orig_df_test = self._rename_varnames(
            df_train, df_test
        )

        self._orig_df_train, self._orig_df_test = self._force_train_test_var_agreement(
            self._orig_df_train, self._orig_df_test
        )

        (
            self._categorical_vars,
            self._numeric_vars,
            self._categorical_to_categories,
        ) = self._compute_categorical_numeric_vars(self._orig_df_train)

        # set the working DataFrames
        self._working_df_train = self._orig_df_train.copy()
        self._working_df_test = self._orig_df_test.copy()

        # keep track of scalers
        self._numeric_var_to_scaler = {var: None for var in self._numeric_vars}

        # set the name
        if name is None:
            self._name = "DataHandler"
        else:
            self._name = name

        # step tracing for all preprocessing steps
        self._preprocess_step_tracer = PreprocessStepTracer()
        self._preprocess_step_tracer.add_category_mapping(
            self._categorical_to_categories
        )

    # --------------------------------------------------------------------------
    # CHECKPOINT HANDLING
    # --------------------------------------------------------------------------
    def load_data_checkpoint(self, checkpoint: str | None = None) -> "DataHandler":
        """The working train and working test DataFrames are reset to the
        original input DataFrames given at object initialization.

        Parameters
        ----------
        checkpoint : str | None
            Default: None. If None, sets the working DataFrames to the original
            DataFrames given at object initialization.

        Returns
        -------
        DataHandler
            Returns self for method chaining.
        """
        if checkpoint is None:
            self._working_df_test = self._orig_df_test.copy()
            self._working_df_train = self._orig_df_train.copy()
            self._numeric_var_to_scaler = {var: None for var in self._numeric_vars}
            self._preprocess_step_tracer = PreprocessStepTracer()
            if self._verbose:
                shapes_dict = self._shapes_str_formatted()
                print_wrapped(
                    "Working DataFrames reset to original DataFrames. "
                    + "Shapes of train, test DataFrames: "
                    + f'{shapes_dict["train"]}, {shapes_dict["test"]}.',
                    type="UPDATE",
                )
        else:
            self._working_df_test = self._checkpoint_name_to_df[checkpoint][0].copy()
            self._working_df_train = self._checkpoint_name_to_df[checkpoint][1].copy()
            self._numeric_var_to_scaler = self._checkpoint_name_to_df[checkpoint][
                2
            ].copy()
            self._preprocess_step_tracer: PreprocessStepTracer = (
                self._checkpoint_name_to_df[checkpoint][3].copy()
            )
            if self._verbose:
                shapes_dict = self._shapes_str_formatted()
                print_wrapped(
                    "Working DataFrames reset to checkpoint "
                    + f'{color_text(checkpoint, "yellow")}. '
                    + "Shapes of train, test DataFrames: "
                    + f'{shapes_dict["train"]}, {shapes_dict["test"]}.',
                    type="UPDATE",
                )
        (
            self._categorical_vars,
            self._numeric_vars,
            self._categorical_to_categories,
        ) = self._compute_categorical_numeric_vars(self._working_df_train)

        return self

    def save_data_checkpoint(self, checkpoint: str) -> "DataHandler":
        """Saves the current state of the working train and test DataFrames.
        The state may be returned to by calling
        load_data_checkpoint(checkpoint).

        Parameters
        ----------
        checkpoint : str
            Name of the checkpoint.

        Returns
        -------
        DataHandler
            Returns self for method chaining.
        """
        if self._verbose:
            print_wrapped(
                "Saved working DataFrames checkpoint "
                + f'{color_text(checkpoint, "yellow")}.',
                type="UPDATE",
            )
        self._checkpoint_name_to_df[checkpoint] = (
            self._working_df_test.copy(),
            self._working_df_train.copy(),
            self._numeric_var_to_scaler.copy(),
            self._preprocess_step_tracer.copy(),
        )

        return self

    def remove_data_checkpoint(self, checkpoint: str) -> "DataHandler":
        """Removes a saved checkpoint to conserve memory.

        Parameters
        ----------
        checkpoint : str
            Name of the checkpoint to remove.

        Returns
        -------
        DataHandler
            Returns self for method chaining.
        """
        out_chkpt = self._checkpoint_name_to_df.pop(checkpoint)
        if self._verbose:
            print_wrapped(
                "Removed working DataFrames checkpoint "
                + f'{color_text(out_chkpt, "yellow")}.',
                type="UPDATE",
            )
        return self

    # --------------------------------------------------------------------------
    # GETTERS
    # --------------------------------------------------------------------------

    def df_all(self) -> pd.DataFrame:
        """Returns the working train and test DataFrames concatenated.

        Returns
        -------
        pd.DataFrame
            Concatenated DataFrames.
        """
        no_test = True
        for a, b in zip(self._working_df_train.index, self._working_df_test.index):
            if a != b:
                no_test = False
                break
        if no_test:
            out = self.df_train()
        else:
            out = pd.concat([self.df_train(), self.df_test()])
        return out

    def df_train(self) -> pd.DataFrame:
        """Returns the working train DataFrame.

        Returns
        -------
        pd.DataFrame
            The working train DataFrame.
        """
        return self._working_df_train

    def df_test(self) -> pd.DataFrame:
        """Returns the working test DataFrame.

        Returns
        -------
        pd.DataFrame
            The working test DataFrame.
        """
        return self._working_df_test

    def vars(self) -> list[str]:
        """Returns a list of all variables in the working DataFrames.

        Returns
        -------
        list[str]
            List of variable names.
        """
        out = self._working_df_train.columns.to_list()
        return out

    def numeric_vars(self) -> list[str]:
        """Returns copy of list of numeric variables.

        Returns
        -------
        list[str]
            List of numeric variable names.
        """
        out = self._numeric_vars.copy()
        return out

    def categorical_vars(self) -> list[str]:
        """Returns copy of list of categorical variables.

        Returns
        -------
        list[str]
            List of categorical variable names.
        """
        out = self._categorical_vars.copy()
        return out

    def scaler(self, var: str) -> BaseSingleVarScaler | None:
        """Returns the scaler for a numeric variable, which could be None.

        Parameters
        ----------
        var : str
            Name of the variable.

        Returns
        -------
        BaseSingleVarScaler | None
            Returns None if there is no scaler for the variable.
        """
        return self._numeric_var_to_scaler[var]

    def train_test_emitter(self, y_var: str, X_vars: list[str]) -> DataEmitter:
        """Returns a DataEmitter object for the working train DataFrame and
        the working test DataFrame.

        Parameters
        ----------
        y_var : str
            Name of the target variable.
        X_vars : list[str]
            Names of the predictor variables.

        Returns
        -------
        DataEmitter
        """
        if y_var not in self._working_df_train.columns:
            raise ValueError(f"Invalid target variable name: {y_var}.")
        for var in X_vars:
            if var not in self._working_df_train.columns:
                raise ValueError(f"Invalid variable name: {var}.")
        return DataEmitter(
            self._orig_df_train,
            self._orig_df_test,
            y_var,
            X_vars,
            self._preprocess_step_tracer,
        )

    def kfold_emitters(
        self,
        y_var: str,
        X_vars: list[str],
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> list[DataEmitter]:
        """Returns a list of DataEmitter objects for cross-validation.
        DataEmitter objects are built from KFold
        (StratifiedKFold if target is categorical) applied to
        the working train DataFrame.

        Parameters
        ----------
        y_var : str
            Name of the target variable.

        X_vars : list[str]
            Names of the predictor variables.

        n_folds : int
            Default: 5. Number of folds.

        shuffle : bool
            Default: True. Whether to shuffle the data.

        random_state : int
            Default: 42. Random state for the
            KFold/StratifiedKFold. Ignored if shuffle is False.

        Returns
        -------
        list[DataEmitter]
        """
        if n_folds < 2:
            raise ValueError("n_folds must be at least 2.")

        if y_var not in self._working_df_train.columns:
            raise ValueError(f"Invalid target variable name: {y_var}.")
        for var in X_vars:
            if var not in self._working_df_train.columns:
                raise ValueError(f"Invalid variable name: {var}.")

        use_stratified = False
        if y_var in self._orig_df_train.columns:
            if self._orig_df_train[y_var].dtype in ["object", "category", "bool"]:
                use_stratified = True
        if use_stratified:
            try:
                if shuffle:
                    kf = StratifiedKFold(
                        n_splits=n_folds, random_state=random_state, shuffle=True
                    )
                else:
                    kf = StratifiedKFold(n_splits=n_folds, shuffle=False)

                out = []
                for train_index, test_index in kf.split(
                    self._orig_df_train, self._orig_df_train[y_var]
                ):
                    df_train = self._orig_df_train.iloc[train_index]
                    df_test = self._orig_df_train.iloc[test_index]
                    out.append(
                        DataEmitter(
                            df_train,
                            df_test,
                            y_var,
                            X_vars,
                            self._preprocess_step_tracer,
                        )
                    )
                return out

            except ValueError as e:
                if self._verbose:
                    print_wrapped(
                        f"StratifiedKFold failed: {e}. Using KFold instead.",
                        type="WARNING",
                    )
                use_stratified = False
                if shuffle:
                    kf = KFold(
                        n_splits=n_folds, random_state=random_state, shuffle=True
                    )
                else:
                    kf = KFold(n_splits=n_folds, shuffle=False)

                out = []
                for train_index, test_index in kf.split(self._orig_df_train):
                    df_train = self._orig_df_train.iloc[train_index]
                    df_test = self._orig_df_train.iloc[test_index]
                    out.append(
                        DataEmitter(
                            df_train,
                            df_test,
                            y_var,
                            X_vars,
                            self._preprocess_step_tracer,
                        )
                    )
                return out

        else:
            if shuffle:
                kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=True)
            else:
                kf = KFold(n_splits=n_folds, shuffle=False)

            out = []
            for train_index, test_index in kf.split(self._orig_df_train):
                df_train = self._orig_df_train.iloc[train_index]
                df_test = self._orig_df_train.iloc[test_index]
                out.append(
                    DataEmitter(
                        df_train, df_test, y_var, X_vars, self._preprocess_step_tracer
                    )
                )
            return out

    # --------------------------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------------------------

    def dropna(
        self,
        include_vars: list[str] | None = None,
        exclude_vars: list[str] | None = None,
    ) -> "DataHandler":
        """Drops rows with missing values in-place on both the working train
        and test DataFrames.

        Parameters
        ----------
        include_vars : list[str]
            Default: None.
            List of columns along which to drop rows with missing values.
            If None, drops rows with missing values in all columns.

        exclude_vars : list[str]
            Default: None.
            List of columns along which to exclude from dropping rows with
            missing values. If None, no variables are excluded.

        Returns
        -------
        DataHandler
            Returns self for method chaining.
        """
        if include_vars is None:
            include_vars = self.vars()
        if exclude_vars is not None:
            include_vars = list(set(include_vars) - set(exclude_vars))

        self._working_df_train = self._working_df_train.dropna(subset=include_vars)
        self._working_df_test = self._working_df_test.dropna(subset=include_vars)
        if self._verbose:
            shapes_dict = self._shapes_str_formatted()
            print_wrapped(
                "Dropped rows with missing values. "
                + "Shapes of train, test DataFrames: "
                + f'{shapes_dict["train"]}, {shapes_dict["test"]}.',
                type="UPDATE",
            )

        self._preprocess_step_tracer.add_step(
            "dropna",
            {
                "vars": include_vars,
            },
        )
        return self

    def onehot(
        self,
        include_vars: list[str] | None = None,
        exclude_vars: list[str] | None = None,
        dropfirst: bool = True,
    ) -> "DataHandler":
        """One-hot encodes all categorical variables in-place. Encoder is
        fit on train DataFrame and transforms both train and test DataFrames.

        Parameters
        ----------
        include_vars : list[str]
            Default: None.
            List of categorical variables to one-hot encode. If None, this is set to
            all categorical variables.

        exclude_vars : list[str]
            Default: None.
            List of categorical variables to exclude from one-hot encoding.

        dropfirst : bool
            Default: True.
            If True, the first dummy variable is dropped.

        Returns
        -------
        DataHandler
            Returns self for method chaining.
        """
        if include_vars is None:
            include_vars = self.categorical_vars()
        if exclude_vars is not None:
            include_vars = list(set(include_vars) - set(exclude_vars))

        self._working_df_train = self._onehot_helper(
            self._working_df_train, vars=include_vars, dropfirst=dropfirst, fit=True
        )
        self._working_df_test = self._onehot_helper(
            self._working_df_test, vars=include_vars, dropfirst=dropfirst, fit=False
        )

        (
            self._categorical_vars,
            self._numeric_vars,
            self._categorical_to_categories,
        ) = self._compute_categorical_numeric_vars(self._working_df_train)

        if self._verbose:
            print_wrapped(
                f"One-hot encoded {list_to_string(include_vars)}. "
                + f"Drop first: {dropfirst}.",
                type="UPDATE",
            )

        self._preprocess_step_tracer.add_step(
            "onehot", {"vars": include_vars, "dropfirst": dropfirst}
        )
        return self

    def drop_highly_missing_vars(self, threshold: float = 0.5) -> "DataHandler":
        """Drops columns with more than 50% missing values
        (computed on train) in-place for both the working train and test
        DataFrames.

        Parameters
        ----------
        threshold : float
            Default: 0.5. Proportion of missing values above which a column is dropped.

        Returns
        -------
        DataHandler
            Returns self for method chaining.
        """
        prev_vars = self._working_df_train.columns.to_list()
        self._working_df_train = self._working_df_train.dropna(
            axis=1, thresh=threshold * len(self._working_df_train)
        )
        curr_vars = self._working_df_train.columns.to_list()
        vars_dropped = set(prev_vars) - set(curr_vars)
        self._working_df_test = self._working_df_test.drop(vars_dropped, axis=1)
        if self._verbose:
            print_wrapped(
                f"Dropped variables {list_to_string(vars_dropped)} "
                + f"with more than {threshold*100}% "
                + "missing values.",
                type="UPDATE",
            )
        (
            self._categorical_vars,
            self._numeric_vars,
            self._categorical_to_categories,
        ) = self._compute_categorical_numeric_vars(self._working_df_train)
        self._preprocess_step_tracer.add_step(
            "drop_highly_missing_vars", {"threshold": threshold}
        )
        return self

    def impute(
        self,
        include_vars: list[str] | None = None,
        exclude_vars: list[str] | None = None,
        numeric_strategy: Literal["median", "mean", "5nn"] = "median",
        categorical_strategy: Literal["most_frequent"] = "most_frequent",
    ) -> "DataHandler":
        """Imputes missing values in-place. Imputer is fit on train DataFrame
        and transforms both train and test DataFrames.

        Parameters
        ----------
        include_vars : list[str]
            Default: None. List of variables to impute missing values.
            If None, imputes missing values in all columns.

        exclude_vars : list[str]
            Default: None. List of variables to exclude from imputing missing values.
            If None, no variables are excluded.

        numeric_strategy : Literal['median', 'mean', '5nn']
            Default: 'median'.
            Strategy for imputing missing values in numeric variables.
            - 'median': impute with median.
            - 'mean': impute with mean.
            - '5nn': impute with 5-nearest neighbors.

        categorical_strategy : Literal['most_frequent']
            Default: 'most_frequent'.
            Strategy for imputing missing values in categorical variables.
            - 'most_frequent': impute with most frequent value.

        Returns
        -------
        DataHandler
            Returns self for method chaining.
        """
        numeric_vars = self.numeric_vars()
        categorical_vars = self.categorical_vars()
        if include_vars is not None:
            include_vars_set = set(include_vars)
            numeric_vars = list(include_vars_set & set(numeric_vars))
            categorical_vars = list(include_vars_set & set(categorical_vars))
        if exclude_vars is not None:
            exclude_vars_set = set(exclude_vars)
            numeric_vars = list(set(numeric_vars) - exclude_vars_set)
            categorical_vars = list(set(categorical_vars) - exclude_vars_set)

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

            if self._verbose:
                print_wrapped(
                    "Imputed missing values with "
                    + "numeric strategy "
                    + f'{color_text(numeric_strategy, "yellow")} and '
                    + "categorical strategy "
                    + f'{color_text(categorical_strategy, "yellow")}.',
                    type="UPDATE",
                )

        self._preprocess_step_tracer.add_step(
            "impute",
            {
                "vars": numeric_vars + categorical_vars,
                "numeric_strategy": numeric_strategy,
                "categorical_strategy": categorical_strategy,
            },
        )

        return self

    def scale(
        self,
        include_vars: list[str] | None = None,
        exclude_vars: list[str] | None = None,
        strategy: Literal["standardize", "minmax", "log", "log1p"] = "standardize",
    ) -> "DataHandler":
        """Scales variable values.

        Parameters
        ----------
        include_vars : list[str]
            Default: None. List of variables to scale.
            If None, scales values in all columns.

        exclude_vars : list[str]
            Default: None. List of variables to exclude from scaling.
            If None, no variables are excluded.

        strategy : Literal["standardize", "minmax", "log", "log1p"]

        Returns
        -------
        DataHandler
            Returns self for method chaining.
        """
        if include_vars is None:
            include_vars = self.numeric_vars()
        if exclude_vars is not None:
            include_vars = list(set(include_vars) - set(exclude_vars))

        for var in include_vars:
            if var not in self._numeric_vars:
                print_wrapped(
                    f"Variable {var} is not numeric. Skipping.", type="WARNING"
                )
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

            if var in self._numeric_var_to_scaler:
                if self._numeric_var_to_scaler[var] is not None:
                    print_wrapped(
                        f"Variable {var} has already been scaled. "
                        "The new scaler will replace the old one.",
                        type="WARNING",
                    )
            self._working_df_train[var] = scaler.transform(
                self._working_df_train[var].to_numpy()
            )
            self._working_df_test[var] = scaler.transform(
                self._working_df_test[var].to_numpy()
            )
            self._numeric_var_to_scaler[var] = scaler

        if self._verbose:
            print_wrapped(
                f"Scaled variables {list_to_string(include_vars)} "
                + f'using strategy {color_text(strategy, "yellow")}.',
                type="UPDATE",
            )

        self._preprocess_step_tracer.add_step(
            "scale", {"vars": include_vars, "strategy": strategy}
        )
        return self

    def add_scaler(self, scaler: BaseSingleVarScaler, var: str) -> "DataHandler":
        """Adds a scaler for the given variable. Does not scale the variable.

        Parameters
        ----------
        scaler : BaseSingleVarScaler
            Scaler object.

        var : str
            Name of the variable.

        Returns
        -------
        DataHandler
            Returns self.
        """
        if self._numeric_var_to_scaler[var] is not None:
            print_wrapped(
                f"Variable {var} has already been scaled. "
                "The new scaler will replace the old one.",
                type="WARNING",
            )
        self._numeric_var_to_scaler[var] = scaler
        self._preprocess_step_tracer.add_step(
            "add_scaler", {"scaler": scaler, "var": var}
        )
        return self

    def select_vars(self, vars: list[str]) -> "DataHandler":
        """Selects subset of (column) variables in-place on the working
        train and test DataFrames.

        Parameters
        ----------
        vars : list[str]
            List of variable names.

        Returns
        -------
        DataHandler
            Returns self.
        """
        self._working_df_test = self._working_df_test[vars]
        self._working_df_train = self._working_df_train[vars]
        (
            self._categorical_vars,
            self._numeric_vars,
            self._categorical_to_categories,
        ) = self._compute_categorical_numeric_vars(self._working_df_train)
        if self._verbose:
            shapes_dict = self._shapes_str_formatted()
            print_wrapped(
                f"Selected columns {list_to_string(vars)}. "
                + "Shapes of train, test DataFrames: "
                + f'{shapes_dict["train"]}, {shapes_dict["test"]}.',
                type="UPDATE",
            )

        self._preprocess_step_tracer.add_step("select_vars", {"vars": vars})

        return self

    def drop_vars(self, vars: list[str]) -> "DataHandler":
        """Drops subset of variables (columns) in-place on the working
        train and test DataFrames.

        Parameters
        ----------
        vars : list[str]

        Returns
        -------
        DataHandler
            Returns self for method chaining.
        """
        self._working_df_test = self._working_df_test.drop(vars, axis="columns")
        self._working_df_train = self._working_df_train.drop(vars, axis="columns")
        (
            self._categorical_vars,
            self._numeric_vars,
            self._categorical_to_categories,
        ) = self._compute_categorical_numeric_vars(self._working_df_train)
        if self._verbose:
            shapes_dict = self._shapes_str_formatted()
            print_wrapped(
                f"Dropped columns {list_to_string(vars)}. "
                + "Shapes of train, test DataFrames: "
                + f'{shapes_dict["train"]}, {shapes_dict["test"]}.',
                type="UPDATE",
            )

        self._preprocess_step_tracer.add_step("drop_vars", {"vars": vars})

        return self

    def force_numeric(self, vars: list[str]) -> "DataHandler":
        """Forces variables to numeric (floats).

        Parameters
        ----------
        vars : list[str]
            Name of variables to force to numeric.

        Returns
        -------
        DataHandler
            Returns self for method chaining.
        """
        for var in vars:
            if var not in self._working_df_train.columns:
                raise ValueError(f"Invalid variable name: {var}.")
            try:
                self._working_df_train[var] = self._working_df_train[var].apply(
                    lambda x: float(x) if pd.notna(x) else np.nan
                )
                self._working_df_test[var] = self._working_df_test[var].apply(
                    lambda x: float(x) if pd.notna(x) else np.nan
                )
            except Exception:
                if self._verbose:
                    print_wrapped(
                        "Unable to force variable "
                        + f'{color_text(var, "purple")} to numeric.',
                        type="WARNING",
                    )

            if self._verbose:
                print_wrapped(
                    f'Forced variable {color_text(var, "purple")} ' + "to numeric.",
                    type="UPDATE",
                )

        self._preprocess_step_tracer.add_step("force_numeric", {"vars": vars})

        return self

    def force_binary(
        self,
        vars: list[str],
        pos_labels: list[str] | None = None,
        ignore_multiclass: bool = False,
        rename: bool = False,
    ) -> "DataHandler":
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
            {pos_label}_yn({var}).

        Returns
        -------
        DataHandler
            Returns self for method chaining.
        """
        if pos_labels is None and ignore_multiclass:
            raise ValueError(
                "pos_labels must be specified if ignore_multiclass is True."
            )

        if not isinstance(vars, list):
            raise ValueError("vars must be lists.")
        if pos_labels is not None and not isinstance(pos_labels, list):
            raise ValueError("pos_labels must be lists.")

        vars_to_renamed = {}
        for i, var in enumerate(vars):
            if var not in self._working_df_train.columns:
                raise ValueError(f"Invalid variable name: {var}.")

            if pos_labels is None:
                unique_vals = self._working_df_train[var].unique()
                if len(unique_vals) > 2:
                    if self._verbose:
                        print_wrapped(
                            "More than two classes present for "
                            + f"{var}. Skipping {var}.",
                            type="WARNING",
                        )
                    continue
                pos_label = unique_vals[0]
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
                        if self._verbose:
                            print_wrapped(
                                "More than two classes present for "
                                + f"{var}. Skipping {var}.",
                                type="WARNING",
                            )
                        continue

                pos_label = pos_labels[i]
                self._working_df_train[var] = self._working_df_train[var].apply(
                    lambda x: 1 if x == pos_label else 0
                )
                self._working_df_test[var] = self._working_df_test[var].apply(
                    lambda x: 1 if x == pos_label else 0
                )

            vars_to_renamed[var] = f"{pos_label}_yn({var})"

        if rename:
            self._working_df_train = self._working_df_train.rename(
                columns=vars_to_renamed
            )
            self._working_df_test = self._working_df_test.rename(
                columns=vars_to_renamed
            )

        if self._verbose:
            if len(vars_to_renamed) == 0:
                print_wrapped("No variables were forced to binary.", type="WARNING")
            else:
                old_vars_txt = color_text(
                    list_to_string(vars_to_renamed.keys()), "purple"
                )
                new_vars_txt = color_text(
                    list_to_string(vars_to_renamed.values()), "purple"
                )
                if rename:
                    print_wrapped(
                        f"Forced variables {old_vars_txt} to binary. "
                        + f"Variables renamed to {new_vars_txt}.",
                        type="UPDATE",
                    )
                else:
                    print_wrapped(
                        f"Forced variables {old_vars_txt} to binary.",
                        type="UPDATE",
                    )

        (
            self._categorical_vars,
            self._numeric_vars,
            self._categorical_to_categories,
        ) = self._compute_categorical_numeric_vars(self._working_df_train)
        self._preprocess_step_tracer.add_step(
            "force_binary",
            {
                "vars": vars,
                "pos_labels": pos_labels,
                "ignore_multiclass": ignore_multiclass,
                "rename": rename,
            },
        )
        return self

    def force_categorical(self, vars: list[str]) -> "DataHandler":
        """Forces variables to become categorical.
        Example use case: create numericly-coded categorical variables.

        Parameters
        ----------
        vars : list[str]

        Returns
        -------
        DataHandler
            Returns self for method chaining.
        """
        if not isinstance(vars, list):
            vars = [vars]

        for var in vars:
            self._working_df_train[var] = self._working_df_train[var].apply(
                lambda x: str(x) if pd.notna(x) else np.nan
            )
            self._working_df_test[var] = self._working_df_test[var].apply(
                lambda x: str(x) if pd.notna(x) else np.nan
            )

        if self._verbose:
            print_wrapped(
                f"Forced variables {list_to_string(vars)} to categorical.",
                type="UPDATE",
            )

        (
            self._categorical_vars,
            self._numeric_vars,
            self._categorical_to_categories,
        ) = self._compute_categorical_numeric_vars(self._working_df_train)

        self._preprocess_step_tracer.add_step("force_categorical", {"vars": vars})

        return self

    # --------------------------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------------------------
    def _verify_input_dfs(self, df1: pd.DataFrame, df2: pd.DataFrame):
        """Ensures that the original train and test DataFrames have the
        same variables.

        Parameters
        ----------
        df1 : pd.DataFrame
            The train DataFrame.

        df2 : pd.DataFrame
            The test DataFrame.

        Raises
        ------
        ValueError
        """
        l1 = df1.columns.to_list()
        l2 = df2.columns.to_list()
        vars_not_in_both = list(set(l1) ^ set(l2))
        if len(vars_not_in_both) > 0:
            raise ValueError(
                f"Variables {list_to_string(vars_not_in_both)} "
                + "are not in both train and test DataFrames."
            )
        datetime_columns = df1.select_dtypes(
            include=["datetime", np.datetime64]
        ).columns.tolist()
        if len(datetime_columns) > 0:
            raise ValueError(
                f"Variables {list_to_string(datetime_columns)} "
                "are of type datetime. TabularMagic cannot handle datetime values."
            )

    def _compute_categorical_numeric_vars(
        self, df: pd.DataFrame
    ) -> tuple[list[str], list[str], dict]:
        """Returns the categorical and numeric column values.
        Also returns the categorical variables mapped to their categories.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        list[str]
            List of categorical variables.

        list[str]
            List of numeric variables.

        dict
            Dictionary mapping categorical variables to their categories.
        """
        categorical_vars = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.to_list()
        numeric_vars = df.select_dtypes(
            exclude=["object", "category", "bool"]
        ).columns.to_list()
        categorical_mapped = self._compute_categories(df, categorical_vars)
        return categorical_vars, numeric_vars, categorical_mapped

    def _force_train_test_var_agreement(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Modifies df_test to have the same columns as df_train. This helps
        mitigate any problems that may arise from one-hot-encoding the test set.

        Parameters
        ----------
        df_train : pd.DataFrame

        df_test : pd.DataFrane

        Returns
        -------
        pd.DataFrame
            Modified train DataFrame.

        pd.DataFrame
            Modified test DataFrame.
        """
        missing_test_columns = list(set(df_train.columns) - set(df_test.columns))
        extra_test_columns = list(set(df_test.columns) - set(df_train.columns))
        if len(extra_test_columns) > 0:
            if self._verbose:
                print_wrapped(
                    f"Columns {list_to_string(extra_test_columns)} not "
                    + "in train have been dropped from test.",
                    type="WARNING",
                )
            df_test = df_test.drop(columns=extra_test_columns, axis=1)
        if len(missing_test_columns) > 0:
            if self._verbose:
                print_wrapped(
                    f"Columns {list_to_string(missing_test_columns)} not "
                    + "in test have been added to test with 0-valued entries.",
                    type="WARNING",
                )
            for col in missing_test_columns:
                df_test[col] = 0
        assert len(df_test.columns) == len(df_train.columns)

        # ensure that the train and test dfs have the same order
        # (for nicer exploration)
        df_test = df_test[df_train.columns]
        for a, b in zip(df_test.columns, df_train.columns):
            assert a == b
        return df_train, df_test

    def _rename_varnames(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        """Renames variables to remove 'problematic' characters.

        We remove the following characters from variable names:
        1. ' ' (whitespace)
        2. '\\n' (newline)
        3. '\\t' (tab)

        Parameters
        ----------
        df_train : pd.DataFrame
            The train DataFrame.

        df_test : pd.DataFrame
            The test DataFrame.

        Returns
        -------
        pd.DataFrame
            The train DataFrame with renamed variable names.

        pd.DataFrame
            The test DataFrame with renamed variable names.
        """
        new_columns = df_train.columns.to_list()

        problematic_chars = [
            " ",
            "\n",
            "\t",
        ]

        orig_renamed_vars = []
        renamed_vars = []

        for i, var in enumerate(new_columns):
            orig_var = var
            for char in problematic_chars:
                if char in var:
                    var = var.replace(char, "")

            if var != orig_var:
                renamed_vars.append(var)
                orig_renamed_vars.append(orig_var)
            new_columns[i] = var

        df_train.columns = new_columns
        df_test.columns = new_columns

        if self._verbose:
            if len(renamed_vars) > 0:
                print_wrapped(
                    f"Renamed variables {list_to_string(orig_renamed_vars)} "
                    + f"to {list_to_string(renamed_vars)}.",
                    type="UPDATE",
                )

        return df_train, df_test

    def _shapes_str_formatted(self) -> dict:
        """Returns a dictionary containing shape information for the
        working DataFrames.

        Returns
        -------
        dict
            {
                "train": shape of working_df_train,
                "test": shape of working_df_test
            }
        """
        return {
            "train": color_text(str(self._working_df_train.shape), color="yellow"),
            "test": color_text(str(self._working_df_test.shape), color="yellow"),
        }

    def _compute_categories(
        self, df: pd.DataFrame, categorical_vars: list[str]
    ) -> dict:
        """Returns a dictionary containing the categorical variables
        each mapped to a list of all categories in the variable.

        Parameters
        ----------
        df : pd.DataFrame

        categorical_vars : list[str]

        Returns
        -------
        dict
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
    ) -> pd.DataFrame:
        """One-hot encodes all categorical variables with more than
        two categories.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to one-hot encode.

        vars : list[str]
            Default: None. If not None, only one-hot encodes the specified variables.

        dropfirst : bool
            Default: True. If True, the first dummy variable is dropped.

        fit : bool
            Default: True.
            If True, fits the encoder on the training data. Otherwise,
            only transforms the test data.

        Returns
        -------
        pd.DataFrame
            The DataFrame with one-hot encoded variables.
        """
        if vars is None:
            categorical_vars, _, _ = self._compute_categorical_numeric_vars(df)
        else:
            print("test", vars)
            print(df.columns)
            for var in vars:
                if var not in df.columns:
                    print("FAIL:", var in df.columns)
                    raise ValueError(f"Invalid variable name: {var}")
            categorical_vars = vars

        if categorical_vars:
            if dropfirst:
                drop = "first"
            else:
                drop = "if_binary"

            if fit:
                self._onehot_encoder = CustomOneHotEncoder(
                    drop=drop, sparse_output=False, handle_unknown="ignore"
                )
                encoded = self._onehot_encoder.fit_transform(df[categorical_vars])
                feature_names = self._onehot_encoder.get_feature_names_out(
                    categorical_vars
                )
                df_encoded = pd.DataFrame(
                    encoded, columns=feature_names, index=df.index
                )

            else:
                encoded = ignore_warnings(self._onehot_encoder.transform)(
                    df[categorical_vars]
                )
                feature_names = self._onehot_encoder.get_feature_names_out(
                    categorical_vars
                )
                df_encoded = pd.DataFrame(
                    encoded, columns=feature_names, index=df.index
                )

            return pd.concat([df_encoded, df.drop(columns=categorical_vars)], axis=1)
        else:
            return df

    def __len__(self) -> int:
        """Returns the number of examples in working_df_train."""
        return len(self._working_df_train)

    def __str__(self) -> str:
        """Returns a string representation of the DataHandler object."""
        working_df_test = self._working_df_test
        working_df_train = self._working_df_train

        max_width = print_options.max_line_width

        textlen_shapes = (
            len(str(working_df_train.shape) + str(working_df_test.shape)) + 25
        )
        shapes_message_buffer_left = (max_width - textlen_shapes) // 2
        shapes_message_buffer_right = int(np.ceil((max_width - textlen_shapes) / 2))

        shapes_message = (
            color_text(bold_text("Train shape: "), "none")
            + color_text(str(working_df_train.shape), "yellow")
            + " " * shapes_message_buffer_left
            + color_text(bold_text("Test shape: "), "none")
            + color_text(str(working_df_test.shape), "yellow")
            + " " * shapes_message_buffer_right
        )

        title_message = color_text(bold_text(self._name), "none")
        title_message = fill_ignore_format(title_message, width=max_width)

        categorical_message = (
            color_text(bold_text("Categorical variables:"), "none") + "\n"
        )

        categorical_vars = self.categorical_vars()
        categorical_var_message = ""
        if len(categorical_vars) == 0:
            categorical_var_message += color_text("None", "yellow")
        else:
            categorical_var_message += list_to_string(categorical_vars)
        categorical_var_message = fill_ignore_format(
            categorical_var_message,
            width=max_width,
            initial_indent=2,
            subsequent_indent=2,
        )

        numeric_message = color_text(bold_text("Numeric variables:"), "none") + "\n"

        numeric_vars = self.numeric_vars()
        numeric_var_message = ""
        if len(numeric_vars) == 0:
            numeric_var_message += color_text("None", "yellow")
        else:
            numeric_var_message += list_to_string(numeric_vars)
        numeric_var_message = fill_ignore_format(
            numeric_var_message,
            width=max_width,
            initial_indent=2,
            subsequent_indent=2,
        )

        bottom_divider = "\n" + color_text("=" * max_width, "none")
        divider = "\n" + color_text("-" * max_width, "none") + "\n"
        divider_invisible = "\n" + " " * max_width + "\n"
        top_divider = color_text("=" * max_width, "none") + "\n"

        final_message = (
            top_divider
            + title_message
            + divider
            + shapes_message
            + divider
            + categorical_message
            + categorical_var_message
            + divider_invisible
            + numeric_message
            + numeric_var_message
            + bottom_divider
        )

        return final_message

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))
