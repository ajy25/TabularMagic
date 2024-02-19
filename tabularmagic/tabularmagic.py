import pandas as pd
from typing import Iterable, Literal
import matplotlib.pyplot as plt
plt.ioff()
from sklearn.model_selection import train_test_split
from .ml import BaseRegression
from .linear import OrdinaryLeastSquares
from .interactive import (ComprehensiveMLRegressionReport, ComprehensiveEDA, 
    FeatureSelectionReport, LinearRegressionReport)
from .preprocessing import DataPreprocessor, RegressionBaseSelector



class TabularMagic():
    """TabularMagic: Automatic statistical and machine learning analysis of 
    datasets in tabular form.
    """

    def __init__(self, df: pd.DataFrame, df_test: pd.DataFrame = None, 
                test_size: float = 0.0, random_state: int = 42):
        """Initializes a TabularMagic object. 
        
        Note: DataFrame indices are not guaranteed to be correctly preserved. 

        Parameters
        ----------
        - df : pd.DataFrame ~ (n_samples, n_variables).
        - df_test : pd.DataFrame ~ (n_test_samples, n_variables).
            Default: None. If not None, then treats df as the train 
            dataset. 
        - test_size : float. 
            Default: 0. Proportion of the dataset to withhold for 
            testing. If test_size = 0, then the train dataset and the 
            test dataset will both be the same as the input df. 
            If df_test is provided, then test_size is ignored. 
        - random_state : int.
            Default: 42. Used for train test split. 
            If df_test is provided, then random_state is ignored. 

        Returns
        -------
        - None
        """
        if df_test is not None:
            self.original_df_train = df.copy()
            self.original_df_test = df_test.copy()
            self.original_df = pd.concat((self.original_df_train, 
                                          self.original_df_test), axis=0)
        else:
            self.original_df = df.copy()
            if test_size > 0:
                temp_train, temp_test = train_test_split(self.original_df, 
                    test_size=test_size, shuffle=True, random_state=random_state)
                self.original_df_train = pd.DataFrame(temp_train, 
                    columns=df.columns)
                self.original_df_test = pd.DataFrame(temp_test, columns=df.columns)
            else:
                self.original_df_train = self.original_df
                self.original_df_test = self.original_df
        self._verify_input_dfs()
        self._dp = None
        self._df_checkpoint_name_to_df = {}
        self.working_df_train = self.original_df_train.copy()
        self.working_df_test = self.original_df_test.copy()
        self.categorical_columns = []
        self.continuous_columns = []
        self._reset_categorical_continuous_vars()




    # --------------------------------------------------------------------------
    # EDA + FEATURE SELECTION + OLS
    # --------------------------------------------------------------------------
        
    def eda(self, dataset: Literal['train', 'test'] = 'train') \
        -> ComprehensiveEDA:
        """Constructs a ComprehensiveEDA object for either the working train 
        or the working test dataset. 

        Parameters
        ----------
        - dataset: Literal['train', 'test'].
            Default: 'train'.

        Returns
        -------
        - ComprehensiveEDA
        """
        if dataset == 'train':
            return ComprehensiveEDA(self.working_df_train)
        elif dataset == 'test':
            return ComprehensiveEDA(self.working_df_test)
        else:
            raise ValueError(f'Invalid input: dataset = {dataset}.')

    def preprocess_data(self, onehot_vars: list[str] = [],
                        standardize_vars: list[str] = [], 
                        minmax_vars: list[str] = [], 
                        imputation_strategy: Literal[None, 'drop', 'mean', 
                        'median', '5nn', '10nn'] = None, 
                        impute_vars_to_skip: list[str] = [],
                        dropfirst_onehot: bool = False):
        """Fits a DataPreprocessor object on the training dataset. Then, 
        preprocesses both the train and test datasets. 

        Note: The working train and test datasets will be modified. 
        
        Parameters
        ----------
        - onehot_vars : list[str]. 
        - standard_scale_vars : list[str].
        - minmax_scale_vars : list[str].
        - imputation_strategy: Literal[None, 'drop', 'mean', 
            'median', '5nn', '10nn'].
        - impute_vars_to_skip: list[str].
        - dropfirst_onehot : bool. 
            Default: False. 
            All binary variables will automatically drop first, 
            regardless of the value of  dropfirst_onehot

        Returns
        -------
        - None
        """
        self._dp = DataPreprocessor(
            self.working_df_train,
            onehot_vars=onehot_vars,
            standardize_vars=standardize_vars,
            minmax_vars=minmax_vars,
            imputation_strategy=imputation_strategy,
            impute_vars_to_skip=impute_vars_to_skip,
            dropfirst_onehot=dropfirst_onehot
        )
        self.working_df_train = self._dp.forward(
            self.working_df_train)
        self.working_df_test = self._dp.forward(
            self.working_df_test)
        self._reset_categorical_continuous_vars()

    def voting_selection(self, X_vars: list[str], y_var: str, 
                         selectors: Iterable[RegressionBaseSelector], 
                         n_target_features: int, 
                         update_working_dfs: bool = False,
                         verbose: bool = True):
        """Supervised feature selection via methods voting based on
        training dataset. 
        Also returns a FeatureSelectionReport object. 
        Can automatically updates the working train and working test
        datasets so that only the selected features remain if 
        update_working_dfs is True.
        
        Parameters
        ----------
        - X_vars : list[str].
            A list of features to look through. 
        - y_var : str.
            The variable to be predicted.
        - selectors : Iterable[BaseSelector].
            Each BaseSelector decides on the top n_target_features.
        - n_target_features : int. 
            Number of desired features, < len(X_vars).
        - update_working_dfs : bool.
            Default: False.
        - verbose : bool.
            Default: True

        Returns
        -------
        - FeatureSelectionReport
        """
        report = FeatureSelectionReport(self.working_df_train, 
            X_vars, y_var, selectors, n_target_features, verbose=verbose)
        if update_working_dfs:
            self.working_df_test = self.working_df_test[report.top_features]
            self.working_df_train = self.working_df_train[report.top_features]
            self._reset_categorical_continuous_vars()
        return report

    def ols(self, X_vars: list[str], y_var: str, 
            regularization_type: Literal[None, 'l1', 'l2'] = None, 
            alpha: float = 0.0):
        """Conducts a simple OLS regression analysis exercise. 

        Parameters
        ----------
        - X_vars : list[str]. 
        - y_var : str. 
        - regularization_type : [None, 'l1', 'l2']. 
            Default: None.
        - alpha : float.
            Default: 0.
        
        Returns
        -------
        - train_report : LinearRegressionReport.
        - test_report : LinearRegressionReport.
        """
        local_X_train_df = self.working_df_train[X_vars]
        local_X_test_df = self.working_df_test[X_vars]
        local_y_train_series = self.working_df_train[y_var]
        local_y_test_series = self.working_df_test[y_var]
        model = OrdinaryLeastSquares(X=local_X_train_df, y=local_y_train_series,
                                     regularization_type=regularization_type,
                                     alpha=alpha)
        model.fit()
        y_var_scaler = None
        if self._dp is not None:
            y_var_scaler = self._dp.get_single_var_scaler(y_var)
        return (
            LinearRegressionReport(model, X_test=local_X_train_df, 
                                   y_test=local_y_train_series,
                                   y_scaler=y_var_scaler),
            LinearRegressionReport(model, X_test=local_X_test_df,
                                   y_test=local_y_test_series,
                                   y_scaler=y_var_scaler),
        )



    # --------------------------------------------------------------------------
    # MACHINE LEARNING
    # --------------------------------------------------------------------------
    def ml_regression_benchmarking(self, X_vars: list[str], y_var: str, 
                                   models: Iterable[BaseRegression], 
                                   verbose: bool = True):
        """Conducts a comprehensive regression benchmarking exercise. 

        Parameters
        ----------
        - X_vars : list[str]. 
        - y_var : str. 
        - models : Iterable[BaseRegression]. 
            Testing performance of all models will be evaluated. 
        - verbose : bool. 
            Default: True. If True, prints progress of tasks (a task is 
            defined as the CV training of one model)
        
        Returns
        -------
        - train_report : ComprehensiveMLRegressionReport.
        - test_report : ComprehensiveMLRegressionReport.
        """
        local_X_train_df = self.working_df_train[X_vars]
        local_X_test_df = self.working_df_test[X_vars]
        local_y_train_series = self.working_df_train[y_var]
        local_y_test_series = self.working_df_test[y_var]
        X_train_np = local_X_train_df.to_numpy()
        y_train_np = local_y_train_series.to_numpy().flatten()
        for i, model in enumerate(models):
            if verbose:
                print(f'Task {i+1} of {len(models)}.\tFitting {model}.')
            model.fit(X_train_np, y_train_np)
        y_var_scaler = None
        if self._dp is not None:
            y_var_scaler = self._dp.get_single_var_scaler(y_var)
        train_report = ComprehensiveMLRegressionReport(
            models, local_X_train_df, local_y_train_series, y_var_scaler)
        test_report = ComprehensiveMLRegressionReport(
            models, local_X_test_df, local_y_test_series, y_var_scaler)
        return train_report, test_report



    # --------------------------------------------------------------------------
    # DATAFRAME MANIPULATION + INDEXING
    # --------------------------------------------------------------------------
    def set_working_df_checkpoint(self, checkpoint: str):
        """Saves the current state of the working train and test datasets. 
        The state may be returned to by calling reset_working_dfs(checkpoint).

        Parameters
        ----------
        - checkpoint : str. 

        Returns
        -------
        - None
        """
        self._df_checkpoint_name_to_df[checkpoint] = (
            self.working_df_train.copy(),
            self.working_df_test.copy()
        )

    def reset_working_dfs(self, checkpoint: str = None):
        """The working train and working test datasets are reset to the 
        input datasets given at object initialization.

        Parameters
        ----------
        - checkpoint : str. 
            Default: None. If None, sets the working datasets to the original 
            datasets given at object initialization. 

        Returns
        _______
        - None
        """
        if checkpoint is None:
            self.working_df_test = self.original_df_test.copy()
            self.working_df_train = self.original_df_train.copy()
        else:
            self.working_df_train =\
                self._df_checkpoint_name_to_df[checkpoint][0].copy()
            self.working_df_test =\
                self._df_checkpoint_name_to_df[checkpoint][1].copy()
        self._reset_categorical_continuous_vars()

    def remove_working_df_checkpoint(self, checkpoint: str):
        """Removes a saved checkpoint to conserve memory.

        Parameters
        ----------
        - checkpoint : str. 

        Returns
        -------
        - None
        """
        self._df_checkpoint_name_to_df.pop(checkpoint)
    
    def select_vars(self, vars: list[str]):
        """Selects subset of (column) variables in-place on the working 
        train and test datasets. 

        Parameters
        ----------
        - vars : list[str]
        """
        self.working_df_train = self.working_df_train[vars]
        self.working_df_test = self.working_df_test[vars]
        self._reset_categorical_continuous_vars()

    def drop_vars(self, vars: list[str]):
        """Drops subset of (column) variables in-place on the working 
        train and test datasets. 

        Parameters
        ----------
        - vars : list[str]
        """
        self.working_df_train = self.working_df_train.drop(vars, axis='columns')
        self.working_df_test = self.working_df_test.drop(vars, axis='columns')
        self._reset_categorical_continuous_vars()


    # --------------------------------------------------------------------------
    # GETTERS
    # --------------------------------------------------------------------------
    def shapes(self):
        """Returns a dictionary containing shape information for the 
        TabularMagic (working) datasets
        
        Returns
        -------
        - dict
        """
        return {
            'working_df_train': self.working_df_train.shape,
            'working_df_test': self.working_df_test.shape
        }

    def retrieve_dfs(self):
        """Returns a tuple (working_df_train, working_df_test). 
        Note that the dataframes are copied before being returned. 
        
        Returns
        -------
        - working_df_train : pd.DataFrame
        - working_df_test : pd.DataFrame
        """
        return self.working_df_train.copy(), self.working_df_test.copy()
    
    def head(self, n = 5):
        """Same as self.working_df_train.head()."""
        return self.working_df_train.head(n)

    def __len__(self):
        """Returns the number of examples in working_df_train.
        """
        return len(self.working_df_train)



    # --------------------------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------------------------
    def _verify_input_dfs(self):
        """Ensures that the original train and test datasets have the 
        same variables. 
        """
        l1 = set(self.original_df_test.columns.to_list())
        l2 = set(self.original_df_train.columns.to_list())
        if len(l1.union(l2)) != len(l1):
            raise RuntimeWarning('The train dataset and test dataset' + \
                ' do not have the same variables.')

    def _reset_categorical_continuous_vars(self):
        self.categorical_columns = self.working_df_train.select_dtypes(
            include=['object', 'category', 'bool']).columns.to_list()
        self.continuous_columns = self.working_df_train.select_dtypes(
            exclude=['object', 'category', 'bool']).columns.to_list()
        




