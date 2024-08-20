import pandas as pd
from typing import Literal
from sklearn.model_selection import train_test_split
from .ml.predict import (
    BaseR,
    MLRegressionReport,
    BaseC,
    MLClassificationReport,
)
from .linear import (
    OLSLinearModel,
    PoissonLinearModel,
    NegativeBinomialLinearModel,
    CountLinearModel,
    BinomialLinearModel,
    BinomialRegressionReport,
    PoissonRegressionReport,
    NegativeBinomialRegressionReport,
    LinearRegressionReport,
    CountRegressionReport,
    parse_and_transform_rlike,
)
from .exploratory import (
    EDAReport,
)
from .display.print_utils import print_wrapped, quote_and_color
from .feature_selection import BaseFSR, BaseFSC
from .data.datahandler import DataHandler
from .utils import ensure_arg_list_uniqueness


class Analyzer:
    """Analyzer is a class designed for conducting exploratory data analysis (EDA),
    regression analysis, and machine learning modeling on wide format tabular data.

    An Analyzer object can be initialized from a single DataFrame which is then
    split into train and test DataFrames, or, alternatively, from pre-split
    train and test DataFrames. The object can then be used to conduct
    a variety of analyses, including exploratory data analysis (the eda() method),
    regression analysis (lm() and glm() methods),
    and machine learning modeling (classify() and regress() methods).

    The Analyzer object also handles data preprocessing tasks, such as scaling,
    imputing missing values, dropping rows with missing values, one-hot encoding,
    and selecting variables. These methods can be chained together for easy data
    transformation. The Analyzer object remembers how the data was transformed,
    enabling proper fitting and transforming of cross validation splits of the
    train dataset.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        df_test: pd.DataFrame | None = None,
        test_size: float = 0.0,
        split_seed: int = 42,
        verbose: bool = True,
        name: str = "Unnamed Dataset",
    ):
        """Initializes a Analyzer object.

        Parameters
        ----------
        df : pd.DataFrame ~ (sample_size, n_variables)
            The DataFrame to be analyzed.

        df_test : pd.DataFrame ~ (test_sample_size, n_variables)
            Default: None.
            If not None, then treats df as the train DataFrame.

        test_size : float
            Default: 0. Proportion of the DataFrame to withhold for
            testing. If test_size = 0, then the train DataFrame and the
            test DataFrame will both be the same as the input df.
            If df_test is provided, then test_size is ignored.

        split_seed : int
            Default: 42.
            Used only for the train test split.
            If df_test is provided, then split_seed is ignored.

        verbose : bool
            Default: False. If True, prints helpful update messages for certain
            Analyzer function calls.

        name : str
            Default: 'Unnamed Dataset'. Name of the dataset the Analyzer is
            initialized for.
        """

        self._verbose = verbose

        df.columns = df.columns.astype(str)

        if df_test is not None:
            df_test.columns = df_test.columns.astype(str)
            self._datahandler = DataHandler(
                df_train=df, df_test=df_test, verbose=self._verbose, name=name
            )

        else:
            if test_size > 0:
                temp_train, temp_test = train_test_split(
                    df, test_size=test_size, shuffle=True, random_state=split_seed
                )
                temp_train_df = pd.DataFrame(temp_train, columns=df.columns)
                temp_test_df = pd.DataFrame(temp_test, columns=df.columns)
            else:
                if self._verbose:
                    print_wrapped(
                        "No test DataFrame provided. The test DataFrame "
                        + "will be treated as a train DataFrame copy.",
                        type="WARNING",
                    )
                temp_train_df = df
                temp_test_df = df
            self._datahandler = DataHandler(
                df_train=temp_train_df,
                df_test=temp_test_df,
                verbose=self._verbose,
                name=name,
            )
        self._name = name

        if self._verbose:
            shapes_dict = self._datahandler._shapes_str_formatted()
            print_wrapped(
                "Analyzer initialized for dataset "
                f"{quote_and_color(self._name, 'yellow')}. "
                + "Shapes of train, test DataFrames: "
                + f'{shapes_dict["train"]}, '
                + f'{shapes_dict["test"]}.',
                type="UPDATE",
            )

    # --------------------------------------------------------------------------
    # EDA + FEATURE SELECTION + REGRESSION ANALYSIS
    # --------------------------------------------------------------------------
    def eda(self, dataset: Literal["train", "test", "all"] = "train") -> EDAReport:
        """Constructs an EDAReport object for the working train
        DataFrame, the working test DataFrame, or both DataFrames combined.

        Parameters
        ----------
        dataset : Literal['train', 'test', 'all']
            Default: 'train'. The dataset to be analyzed.

        Returns
        -------
        EDAReport
            The EDAReport object contains a variety of exploratory data
            analysis methods, including summary statistics for numeric and
            categorical variables, t-tests, and data visualizations.
        """
        if dataset == "train":
            return EDAReport(self._datahandler.df_train())
        elif dataset == "test":
            return EDAReport(self._datahandler.df_test())
        elif dataset == "all":
            return EDAReport(self._datahandler.df_all())
        else:
            raise ValueError(f"Invalid input: dataset = {dataset}.")

    @ensure_arg_list_uniqueness()
    def lm(
        self,
        target: str | None = None,
        predictors: list[str] | None = None,
        formula: str | None = None,
    ) -> LinearRegressionReport:
        """Conducts a simple OLS regression analysis exercise.
        If formula is provided, performs regression with OLS via formula.
        Examples with missing data will be dropped.

        Parameters
        ----------
        target : str
            Default: None. The variable to be predicted.

        predictors : list[str]
            Default: None.
            If None, all variables except target will be used as predictors.

        formula : str
            Default: None. If not None, uses formula to specify the regression
            (overrides target and predictors).

        Returns
        -------
        LinearRegressionReport
        """
        if formula is None and target is None:
            raise ValueError("target must be specified if formula is None.")

        elif formula is None:
            if predictors is None:
                predictors = self._datahandler.vars()
                if target in predictors:
                    predictors.remove(target)
            return LinearRegressionReport(
                OLSLinearModel(), self._datahandler, target, predictors
            )

        else:
            try:
                y_series_train, y_scaler, X_df_train = parse_and_transform_rlike(
                    formula, self._datahandler.df_train()
                )
                y_series_test, _, X_df_test = parse_and_transform_rlike(
                    formula, self._datahandler.df_test()
                )
            except Exception as e:
                raise ValueError(f"Invalid formula: {formula}. Error: {e}.")

            # ensure missing values are dropped
            y_X_df_combined_train = pd.DataFrame(y_series_train).join(X_df_train)
            y_X_df_combined_test = pd.DataFrame(y_series_test).join(X_df_test)
            y_X_df_combined_train = y_X_df_combined_train.dropna()
            y_X_df_combined_test = y_X_df_combined_test.dropna()
            (
                y_X_df_combined_train,
                y_X_df_combined_test,
            ) = self._datahandler._force_train_test_var_agreement(
                y_X_df_combined_train, y_X_df_combined_test
            )

            predictors = y_X_df_combined_train.columns.to_list()
            target = y_series_train.name
            predictors.remove(target)

            datahandler = DataHandler(
                y_X_df_combined_train, y_X_df_combined_test, verbose=False
            )
            if y_scaler is not None:
                datahandler.add_scaler(y_scaler, target)
            elif self._datahandler.scaler(target) is not None:
                datahandler.add_scaler(self._datahandler.scaler(target), target)

            return LinearRegressionReport(
                OLSLinearModel(), datahandler, target, predictors
            )

    @ensure_arg_list_uniqueness()
    def glm(
        self,
        family: Literal["poisson", "binomial", "negbinomial", "count"],
        target: str | None = None,
        predictors: list[str] | None = None,
        formula: str | None = None,
    ) -> (
        PoissonRegressionReport
        | BinomialRegressionReport
        | NegativeBinomialRegressionReport
        | CountRegressionReport
    ):
        """Conducts a generalized linear regression exercise.
        If formula is provided, performs linear regression with link
        function depending on specified family via formula.
        Examples with missing data will be dropped.

        Parameters
        ----------
        family : Literal["poisson", "binomial",  "negbinomial", "count"]
            The family of the GLM.

        target : str
            Default: None. The variable to be predicted.

        predictors : list[str]
            Default: None. If None, all variables except target will be used as
            predictors.

        formula : str
            Default: None. If not None, uses formula to specify the regression
            (overrides target and predictors).

        Returns
        -------
        PoissonRegressionReport | BinomialRegressionReport |
        NegativeBinomialRegressionReport | CountRegressionReport
            The appropriate regression report object is returned
            depending on the specified family.
        """
        if formula is None and target is None:
            raise ValueError("target must be specified if formula is None.")

        elif formula is None:
            if predictors is None:
                predictors = self._datahandler.vars()
                if target in predictors:
                    predictors.remove(target)
            if family == "poisson":
                return PoissonRegressionReport(
                    PoissonLinearModel(), self._datahandler, target, predictors
                )
            elif family == "binomial":
                return BinomialRegressionReport(
                    BinomialLinearModel(), self._datahandler, target, predictors
                )
            elif family == "negbinomial":
                return NegativeBinomialRegressionReport(
                    NegativeBinomialLinearModel(), self._datahandler, target, predictors
                )
            elif family == "count":
                return CountRegressionReport(
                    CountLinearModel(), self._datahandler, target, predictors
                )
            else:
                raise ValueError("invalid input for family")

        else:
            try:
                y_series_train, y_scaler, X_df_train = parse_and_transform_rlike(
                    formula, self._datahandler.df_train()
                )
                y_series_test, _, X_df_test = parse_and_transform_rlike(
                    formula, self._datahandler.df_test()
                )
            except Exception as e:
                raise ValueError(f"Invalid formula: {formula}. " f"Error: {e}.")

            # ensure missing values are dropped
            y_X_df_combined_train = pd.DataFrame(y_series_train).join(X_df_train)
            y_X_df_combined_test = pd.DataFrame(y_series_test).join(X_df_test)
            y_X_df_combined_train = y_X_df_combined_train.dropna()
            y_X_df_combined_test = y_X_df_combined_test.dropna()
            (
                y_X_df_combined_train,
                y_X_df_combined_test,
            ) = self._datahandler._force_train_test_var_agreement(
                y_X_df_combined_train, y_X_df_combined_test
            )

            predictors = y_X_df_combined_train.columns.to_list()
            target = y_series_train.name
            predictors.remove(target)

            datahandler = DataHandler(
                y_X_df_combined_train, y_X_df_combined_test, verbose=False
            )
            datahandler.add_scaler(y_scaler, target)

            if family == "poisson":
                return PoissonRegressionReport(
                    PoissonLinearModel(), self._datahandler, target, predictors
                )
            elif family == "binomial":
                return BinomialRegressionReport(
                    BinomialLinearModel(), self._datahandler, target, predictors
                )
            elif family == "negbinomial":
                return NegativeBinomialRegressionReport(
                    NegativeBinomialLinearModel(), self._datahandler, target, predictors
                )
            elif family == "count":
                return CountRegressionReport(
                    CountLinearModel(), self._datahandler, target, predictors
                )
            else:
                raise ValueError("invalid input for family")

    # --------------------------------------------------------------------------
    # MACHINE LEARNING
    # --------------------------------------------------------------------------

    @ensure_arg_list_uniqueness()
    def regress(
        self,
        models: list[BaseR],
        target: str,
        predictors: list[str] | None = None,
        feature_selectors: list[BaseFSR] | None = None,
        max_n_features: int | None = None,
        outer_cv: int | None = None,
        outer_cv_seed: int = 42,
    ) -> MLRegressionReport:
        """Conducts a comprehensive regression ML model benchmarking exercise.
        Observations with missing data will be dropped.

        Parameters
        ----------
        models : list[BaseR]
            Models to be evaluated.

        target : str
            The variable to be predicted.

        predictors : list[str]
            Default: None.
            If None, uses all variables except target as predictors.

        feature_selectors : list[BaseFSR]
            The feature selectors for voting selection. Feature selectors
            can be used to select the most important predictors.
            Feature selectors can also be specified at the model level. If
            specified here, the same feature selectors will be used for all
            models.

        max_n_features : int
            Default: None. Maximum number of predictors to utilize.
            Ignored if feature_selectors is None.
            If None, then all features with at least 50% support are selected.

        outer_cv : int
            Default: None. If not None, reports training scores via nested k-fold CV.

        outer_cv_seed : int
            Default: 42. The random seed for the outer cross validation loop.

        Returns
        -------
        MLRegressionReport
        """
        if predictors is None:
            predictors = self._datahandler.vars()
            if target in predictors:
                predictors.remove(target)

        return MLRegressionReport(
            models=models,
            datahandler=self._datahandler,
            target=target,
            predictors=predictors,
            feature_selectors=feature_selectors,
            max_n_features=max_n_features,
            outer_cv=outer_cv,
            outer_cv_seed=outer_cv_seed,
            verbose=self._verbose,
        )

    @ensure_arg_list_uniqueness()
    def classify(
        self,
        models: list[BaseC],
        target: str,
        predictors: list[str] | None = None,
        feature_selectors: list[BaseFSC] | None = None,
        max_n_features: int | None = None,
        outer_cv: int | None = None,
        outer_cv_seed: int = 42,
    ) -> MLClassificationReport:
        """Conducts a comprehensive classification ML model benchmarking exercise.
        Observations with missing data will be dropped.

        Parameters
        ----------
        models : list[BaseC]
            Models to be evaluated.

        target : str
            The variable to be predicted.

        predictors : list[str]
            Default: None.
            If None, uses all variables except target as predictors.

        feature_selectors : list[BaseFSR]
            The feature selectors for voting selection. Feature selectors
            can be used to select the most important predictors.
            Feature selectors can also be specified at the model level. If
            specified here, the same feature selectors will be used for all
            models.

        max_n_features : int
            Default: None.
            Maximum number of predictors to utilize.
            Ignored if feature_selectors is None.
            If None, then all features with at least 50% support are selected.

        outer_cv : int
            Default: None.
            If not None, reports training scores via nested k-fold CV.

        outer_cv_seed : int
            Default: 42.
            The random seed for the outer cross validation loop.

        Returns
        -------
        MLClassificationReport
        """
        if predictors is None:
            predictors = self._datahandler.vars()
            if target in predictors:
                predictors.remove(target)

        return MLClassificationReport(
            models=models,
            datahandler=self._datahandler,
            target=target,
            predictors=predictors,
            feature_selectors=feature_selectors,
            max_n_features=max_n_features,
            outer_cv=outer_cv,
            outer_cv_seed=outer_cv_seed,
            verbose=self._verbose,
        )

    # --------------------------------------------------------------------------
    # DATAHANDLER METHODS
    # --------------------------------------------------------------------------
    def load_data_checkpoint(self, checkpoint_name: str | None = None) -> "Analyzer":
        """Loads the original train and test DataFrames.

        Parameters
        ----------
        checkpoint_name : str
            Default: None. The name of the checkpoint to load.
            If None, loads the original train and test DataFrames.

        Returns
        -------
        Analyzer
            Returns self for method chaining.
        """
        self._datahandler.load_data_checkpoint(checkpoint_name)
        return self

    def save_data_checkpoint(self, checkpoint_name: str) -> "Analyzer":
        """Saves the current train and test DataFrames.

        Parameters
        ----------
        checkpoint_name : str
            The name of the checkpoint.

        Returns
        -------
        Analyzer
            Returns self for method chaining.
        """
        self._datahandler.save_data_checkpoint(checkpoint_name)
        return self

    def remove_data_checkpoint(self, checkpoint_name: str) -> "Analyzer":
        """Deletes a saved checkpoint.

        Parameters
        ----------
        checkpoint_name : str
            The name of the checkpoint to delete.

        Returns
        -------
        Analyzer
            Returns self for method chaining.
        """
        self._datahandler.remove_data_checkpoint(checkpoint_name)
        return self

    @ensure_arg_list_uniqueness()
    def scale(
        self,
        include_vars: list[str] | None = None,
        exclude_vars: list[str] | None = None,
        strategy: Literal["standardize", "minmax", "log", "log1p"] = "standardize",
    ) -> "Analyzer":
        """Scales the variables.

        Parameters
        ----------
        include_vars : list[str]
            Default: None. List of variables to scale.
            If None, scales values in all columns.

        exclude_vars : list[str]
            Default: None. List of variables to exclude from scaling.
            If None, no variables are excluded.

        strategy : Literal["standardize", "minmax", "log", "log1p"]
            Default: 'standardize'. The scaling strategy.

        Returns
        -------
        Analyzer
            Returns self for method chaining.
        """
        self._datahandler.scale(
            include_vars=include_vars,
            exclude_vars=exclude_vars,
            strategy=strategy,
        )
        return self

    @ensure_arg_list_uniqueness()
    def impute(
        self,
        include_vars: list[str] | None = None,
        exclude_vars: list[str] | None = None,
        numeric_strategy: Literal["median", "mean", "5nn"] = "median",
        categorical_strategy: Literal["most_frequent"] = "most_frequent",
    ) -> "Analyzer":
        """Imputes missing values. The imputer is fit on the train DataFrame
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
        Analyzer
            Returns self for method chaining.
        """
        self._datahandler.impute(
            include_vars=include_vars,
            exclude_vars=exclude_vars,
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy,
        )
        return self

    @ensure_arg_list_uniqueness()
    def dropna(
        self,
        include_vars: list[str] | None = None,
        exclude_vars: list[str] | None = None,
    ) -> "Analyzer":
        """Drops observations (rows) with missing values on both the train
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
        Analyzer
            Returns self for method chaining.
        """
        self._datahandler.dropna(
            include_vars=include_vars,
            exclude_vars=exclude_vars,
        )
        return self

    def drop_highly_missing_vars(self, threshold: float = 0.5) -> "Analyzer":
        """Drops variables (columns) with missingness rate above a specified threshold.

        Parameters
        ----------
        threshold : float
            Default: 0.5. The threshold above which variables are dropped.

        Returns
        -------
        Analyzer
            Returns self for method chaining.
        """
        self._datahandler.drop_highly_missing_vars(threshold)
        return self

    @ensure_arg_list_uniqueness()
    def onehot(
        self,
        include_vars: list[str] | None = None,
        exclude_vars: list[str] | None = None,
        dropfirst: bool = True,
    ) -> "Analyzer":
        """One-hot encodes the specified variables (columns).

        Parameters
        ----------
        include_vars : list[str]
            Default: None. List of variables to one-hot encode.
            If None, one-hot encodes all categorical variables.

        exclude_vars : list[str]
            Default: None. List of variables to exclude from one-hot encoding.
            If None, no variables are excluded.

        dropfirst : bool
            Default: True. If True, drops the first one-hot encoded column.

        Returns
        -------
        Analyzer
            Returns self for method chaining.
        """
        self._datahandler.onehot(
            include_vars=include_vars, exclude_vars=exclude_vars, dropfirst=dropfirst
        )
        return self

    @ensure_arg_list_uniqueness()
    def select_vars(
        self,
        include_vars: list[str] | None = None,
        exclude_vars: list[str] | None = None,
    ) -> "Analyzer":
        """Selects the specified variables.

        Parameters
        ----------
        include_vars : list[str]
            Default: None. List of variables to include.
            If None, includes all variables.

        exclude_vars : list[str]
            Default: None. List of variables to exclude.
            If None, no variables are excluded.

        Returns
        -------
        Analyzer
            Returns self for method chaining.
        """
        if include_vars is None:
            include_vars = self._datahandler.vars()
        if exclude_vars is not None:
            include_vars = list(set(include_vars) - set(exclude_vars))
        self._datahandler.select_vars(vars=include_vars)
        return self

    @ensure_arg_list_uniqueness()
    def force_numeric(self, vars: list[str]) -> "Analyzer":
        """Forces specificed variables to numeric (float).

        Parameters
        ----------
        vars : list[str]
            Name of variables to force to numeric.

        Returns
        -------
        Analyzer
            Returns self for method chaining.
        """
        self._datahandler.force_numeric(vars)
        return self

    @ensure_arg_list_uniqueness()
    def force_categorical(self, vars: list[str]) -> "Analyzer":
        """Forces specificed variables (columns) to have categorical values.
        That is, the variables' values are converted to strings.

        Parameters
        ----------
        vars : list[str]
            Name of variables to force to categorical.

        Returns
        -------
        Analyzer
            Returns self for method chaining.
        """
        self._datahandler.force_categorical(vars)
        return self

    @ensure_arg_list_uniqueness()
    def force_binary(
        self,
        vars: list[str],
        pos_labels: list[str] | None = None,
        ignore_multiclass: bool = False,
        rename: bool = False,
    ) -> "Analyzer":
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
        Analyzer
            Returns self for method chaining.
        """
        self._datahandler.force_binary(
            vars=vars,
            pos_labels=pos_labels,
            ignore_multiclass=ignore_multiclass,
            rename=rename,
        )
        return self

    def datahandler(self) -> DataHandler:
        """Returns the DataHandler.

        Returns
        -------
        DataHandler
            The DataHandler object takes care of data management and preprocessing.
        """
        return self._datahandler

    # --------------------------------------------------------------------------
    # MAGIC METHODS
    # --------------------------------------------------------------------------
    def __len__(self) -> int:
        """Returns the number of examples in working train DataFrame."""
        return len(self._datahandler)

    def __str__(self) -> str:
        """Returns metadata in string form."""
        return self._datahandler.__str__()

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))
