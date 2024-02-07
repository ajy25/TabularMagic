import pandas as pd
from typing import Iterable, Literal
from sklearn.model_selection import train_test_split
from .ml_models import *
from .interactive import (MLRegressionReport, ComprehensiveEDA, 
    FeatureSelectionReport)
from .preprocessing import DataPreprocessor, RegressionBaseSelector


class TabularMagic():
    """TabularMagic: Automatic statistical and machine learning analysis of 
    datasets in tabular form.
    """

    def __init__(self, df: pd.DataFrame, df_test: pd.DataFrame = None, 
                test_size: float = 0.2, random_state: int = 42):
        """Initializes a TabularMagic object. 
        
        Note: DataFrame indices are not guaranteed to be correctly preserved. 

        Parameters
        ----------
        - df : pd.DataFrame ~ (n_samples, n_variables).
        - df_test : pd.DataFrame ~ (n_test_samples, n_variables).
            Default: None. If not None, then treats df as the train 
            dataset. 
        - test_size : float. 
            Default: 0.2. Proportion of the dataset to withhold for 
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

        # Exploratory Data Analysis on raw input data
        self.train_eda = ComprehensiveEDA(self.original_df_train)
        self.test_eda = ComprehensiveEDA(self.original_df_test)

        self.working_df_train = self.original_df_train.copy()
        self.working_df_test = self.original_df_test.copy()

        self._dp = None


    def preprocess_data(self, onehot_vars: list[str] = [],
                        standardize_vars: list[str] = [], 
                        minmax_vars: list[str] = [], 
                        imputation_strategy: Literal[None, 'drop', 'mean', 
                        'median', '5nn', '10nn'] = None, 
                        dropfirst_onehot: bool = False):
        """Fits a DataPreprocessor object on the training dataset. Then, 
        preprocesses both the training and testing datasets. 
        
        Parameters
        ----------
        - onehot_vars : list[str]. 
        - standard_scale_vars : list[str].
        - minmax_scale_vars : list[str].
        - imputation_strategy: Literal[None, 'drop', 'mean', 
            'median', '5nn', '10nn'].
        - dropfirst_onehot : bool. 

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
            dropfirst_onehot=dropfirst_onehot
        )
        self.working_df_train = self._dp.forward(
            self.working_df_train)
        self.working_df_test = self._dp.forward(
            self.working_df_test)


    def voting_selection(self, X_vars: list[str], y_var: str, 
                         selectors: Iterable[RegressionBaseSelector], 
                         n_target_features: int, 
                         update_working_dfs: bool = False):
        """Supervised feature selection via methods voting based on
        training dataset. 
        Also returns a FeatureSelectionReport object. 
        Can automatically updates the working test/train 
        DataFrames so that only the selected features remain if 
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

        Returns
        -------
        - FeatureSelectionReport
        """
        report = FeatureSelectionReport(self.working_df_train, 
            X_vars, y_var, selectors, n_target_features)
        
        if update_working_dfs:
            self.working_df_test = self.working_df_test[report.top_features]
            self.working_df_train = self.working_df_train[report.top_features]

        return report



    def ml_regression_benchmarking(self, X_vars: list[str], y_var: str, 
                                   models: Iterable[BaseModel]):
        """Conducts a comprehensive regression benchmarking exercise. 

        Parameters
        ----------
        - X_vars : list[str]. 
        - y_var : str. 
        - models : Iterable[BaseModel]. 
            Testing performance of all models will be evaluated. 

        Returns
        -------
        - train_report : MLRegressionReport.
        - test_report : MLRegressionReport.
        """
        self._X_vars = X_vars
        self._y_var = y_var
        self._models = models

        local_X_train_df = self.working_df_train[X_vars]
        local_X_test_df = self.working_df_test[X_vars]
        local_y_train_df = self.working_df_train[y_var]
        local_y_test_df = self.working_df_test[y_var]

        X_train_np = local_X_train_df.to_numpy()
        y_train_np = local_y_train_df.to_numpy().flatten()

        for i, model in enumerate(models):
            print(f'Task {i+1} of {len(models)}.\tTraining {model}.')
            model.fit(X_train_np, y_train_np)


        y_var_scaler = None
        if self._dp is not None:
            y_var_scaler = self._dp.get_single_var_scaler(y_var)

        train_report = MLRegressionReport(
            models, local_X_train_df, local_y_train_df, y_var_scaler)
        test_report = MLRegressionReport(
            models, local_X_test_df, local_y_test_df, y_var_scaler)

        return train_report, test_report


    def reset_working_dfs(self):
        """The working DataFrames are reset to the DataFrames given at object 
        initialization.
        """
        self.working_df_test = self.original_df_test.copy()
        self.working_df_train = self.original_df_train.copy()


    
    def _verify_input_dfs(self):
        l1 = set(self.original_df_test.columns.to_list())
        l2 = set(self.original_df_train.columns.to_list())
        if len(l1.union(l2)) != len(l1):
            raise RuntimeWarning('The train DataFrame and test DataFrame' + \
                ' do not have the same variables.')



