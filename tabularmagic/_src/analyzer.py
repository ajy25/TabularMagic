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
    ComprehensiveEDA,
)
from .display.print_utils import print_wrapped
from .feature_selection import BaseFSR, BaseFSC
from .data.datahandler import DataHandler


class Analyzer:
    """Analyzer: A class designed for conducting exploratory data analysis (EDA),
    regression analysis, and machine learning modeling on wide format tabular data.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        df_test: pd.DataFrame | None = None,
        test_size: float = 0.0,
        split_seed: int = 42,
        verbose: bool = True,
        name: str = "Analyzer",
    ):
        """Initializes a Analyzer object.

        Parameters
        ----------
        df : pd.DataFrame ~ (sample_size, n_variables).
            The DataFrame to be analyzed.
        df_test : pd.DataFrame ~ (test_sample_size, n_variables).
            Default: None.
            If not None, then treats df as the train DataFrame.
        test_size : float.
            Default: 0. Proportion of the DataFrame to withhold for
            testing. If test_size = 0, then the train DataFrame and the
            test DataFrame will both be the same as the input df.
            If df_test is provided, then test_size is ignored.
        split_seed : int.
            Default: 42.
            Used only for the train test split.
            If df_test is provided, then split_seed is ignored.
        verbose : bool.
            Default: False. If True, prints helpful update messages for certain
                Analyzer function calls.
        name : str.
            Default: 'Analyzer'. Identifier for object.
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
                "Analyzer initialized. "
                + "Shapes of train, test DataFrames: "
                + f'{shapes_dict["train"]}, '
                + f'{shapes_dict["test"]}.',
                type="UPDATE",
            )

    # --------------------------------------------------------------------------
    # EDA + FEATURE SELECTION + REGRESSION ANALYSIS
    # --------------------------------------------------------------------------
    def eda(
        self, dataset: Literal["train", "test", "all"] = "train"
    ) -> ComprehensiveEDA:
        """Constructs a ComprehensiveEDA object for the working train
        DataFrame, the working test DataFrame, or both DataFrames combined.

        Parameters
        ----------
        dataset: Literal['train', 'test', 'all'].
            Default: 'train'.

        Returns
        -------
        ComprehensiveEDA
        """
        if dataset == "train":
            return ComprehensiveEDA(self._datahandler.df_train())
        elif dataset == "test":
            return ComprehensiveEDA(self._datahandler.df_test())
        elif dataset == "all":
            return ComprehensiveEDA(self._datahandler.df_all())
        else:
            raise ValueError(f"Invalid input: dataset = {dataset}.")

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
        target : str.
            Default: None. The variable to be predicted.
        predictors : list[str].
            Default: None.
            If None, all variables except target will be used as predictors.
        formula : str.
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

            return LinearRegressionReport(
                OLSLinearModel(), datahandler, target, predictors
            )

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
        family : Literal["poisson", "binomial"].
            The family of the GLM.
        target : str.
            Default: None. The variable to be predicted.
        predictors : list[str].
            Default: None.
            If None, all variables except target will be used as predictors.
        formula : str.
            Default: None. If not None, uses formula to specify the regression
            (overrides target and predictors).

        Returns
        -------
        PoissonRegressionReport or BinomialRegressionReport
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
        """Conducts a comprehensive regression benchmarking exercise.
        Examples with missing data will be dropped.

        Parameters
        ----------
        models : list[BaseRegression].
            Testing performance of all models will be evaluated.
        target : str.
        predictors : list[str].
            Default: None.
            If None, uses all variables except target as predictors.
        feature_selectors : list[BaseFSR].
            The feature selectors for voting selection. Feature selectors
            can be used to select the most important predictors.
            Feature selectors can also be specified at the model level. If
            specified here, the same feature selectors will be used for all
            models.
        max_n_features : int.
            Default: None.
            Maximum number of predictors to utilize. 
            Ignored if feature_selectors is None.
            If None, then all features with at least 50% support are selected.
        outer_cv : int.
            Default: None.
            If not None, reports training scores via nested k-fold CV.
        outer_cv_seed : int.
            Default: 42.
            The random seed for the outer cross validation loop.

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
        """Conducts a comprehensive classification benchmarking exercise.
        Examples with missing data will be dropped.

        Parameters
        ----------
        models : list[BaseClassification].
            Testing performance of all models will be evaluated.
        target : str.
        predictors : list[str].
            Default: None.
            If None, uses all variables except target as predictors.
        feature_selectors : list[BaseFSR].
            The feature selectors for voting selection. Feature selectors
            can be used to select the most important predictors.
            Feature selectors can also be specified at the model level. If
            specified here, the same feature selectors will be used for all
            models.
        max_n_features : int.
            Default: None.
            Maximum number of predictors to utilize. 
            Ignored if feature_selectors is None.
            If None, then all features with at least 50% support are selected.
        outer_cv : int.
            Default: None.
            If not None, reports training scores via nested k-fold CV.
        outer_cv_seed : int.
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
    # GETTERS
    # --------------------------------------------------------------------------
    def datahandler(self) -> DataHandler:
        """Returns the DataHandler."""
        return self._datahandler

    def __len__(self):
        """Returns the number of examples in working train DataFrame."""
        return len(self._datahandler)

    def __str__(self):
        """Returns metadata in string form."""
        return self._datahandler.__str__()

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))
