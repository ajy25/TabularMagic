import math
import pandas as pd
from typing import Iterable, Literal
import matplotlib.pyplot as plt
plt.ioff()
from textwrap import fill
from sklearn.model_selection import train_test_split

from .ml.discriminative.regression.base_regression import BaseRegression
from .linear.regression.linear_regression import OrdinaryLeastSquares
from .linear.regression.lm_rlike_util import parse_and_transform_rlike
from .interactive import (ComprehensiveMLRegressionReport, ComprehensiveEDA, 
    RegressionVotingSelectionReport, LinearRegressionReport)
from .util.console import print_wrapped, color_text
from .util.constants import TOSTR_MAX_WIDTH
from .feature_selection import RegressionBaseSelector
from .preprocessing.datapreprocessor import DataPreprocessor



class TabularMagic:
    """TabularMagic: Automatic statistical and machine learning analysis of 
    datasets in tabular form. 
    """

    def __init__(self, df: pd.DataFrame, df_test: pd.DataFrame = None, 
            test_size: float = 0.0, split_seed: int = 42, 
            verbose: bool = True, id: str = 'TabularMagic'):
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
        - split_seed : int.
            Default: 42. Used only for the train test split. 
            If df_test is provided, then split_seed is ignored. 
        - verbose : bool. 
            Default: False. If True, prints helpful update messages for certain 
                TabularMagic function calls.
        - id : str. 
            Identifier for object. 

        Returns
        -------
        - None
        """
        self._tm_verbose = verbose
        if df_test is not None:
            self._original_df_train = df.copy()
            self._original_df_test = df_test.copy()
            self._original_df = pd.concat((self._original_df_train, 
                                          self._original_df_test), axis=0)
        else:
            self._original_df = df.copy()
            if test_size > 0:
                temp_train, temp_test = train_test_split(self._original_df, 
                    test_size=test_size, shuffle=True, 
                    random_state=split_seed)
                self._original_df_train = pd.DataFrame(temp_train, 
                    columns=df.columns)
                self._original_df_test = pd.DataFrame(temp_test, 
                                                     columns=df.columns)
            else:
                if self._tm_verbose:
                    print_wrapped(f'{color_text("WARNING:", "red")} ' +\
                          'No test dataset provided. ' +\
                          'Test dataset will be treated as train dataset copy.')
                self._original_df_train = self._original_df
                self._original_df_test = self._original_df
        self._verify_input_dfs()
        self._dp = None
        self._df_checkpoint_name_to_df = {}
        self._working_df_train = self._original_df_train.copy()
        self._working_df_test = self._original_df_test.copy()
        self._categorical_vars = []
        self._continuous_vars = []
        self._id = id
        self._remove_spaces_varnames()
        self._reset_categorical_continuous_vars()

        if self._tm_verbose:
            shapes_dict = self.shapes()
            print_wrapped(f'{color_text("UPDATE:", "green")} TabularMagic' +\
                   ' initialization complete. ' +\
                'Shapes of train, test datasets: ' + \
                f'{shapes_dict["train"]}, {shapes_dict["test"]}')


    # --------------------------------------------------------------------------
    # EDA + FEATURE SELECTION + REGRESSION ANALYSIS
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
            return ComprehensiveEDA(self._working_df_train)
        elif dataset == 'test':
            return ComprehensiveEDA(self._working_df_test)
        else:
            raise ValueError(f'Invalid input: dataset = {dataset}.')


    def preprocess_data(self, 
                        imputation_strategy: Literal[None, 'drop', 
                            'median-mostfrequent', 'mean-mostfrequent', 
                            '5nn-mostfrequent'] = None,
                        onehot_vars: list[str] = [],
                        standardize_vars: list[str] = [], 
                        minmax_vars: list[str] = [], 
                        log1p_vars: list[str] = [],
                        log_vars: list[str] = [], 
                        dropfirst_onehot: bool = False):
        """Fits a DataPreprocessor object on the training dataset. Then, 
        preprocesses both the train and test datasets. 

        Note: The working train and test datasets will be modified. 
        
        Parameters
        ----------
        - imputation_strategy : Literal[None, 'drop', 
            'median-mostfrequent', 'mean-mostfrequent', 
            '5nn-mostfrequent']. 
            Imputation strategy described for 
            continuous-categorical variables. 
        - onehot_vars : list[str]. 
        - standard_scale_vars : list[str].
        - minmax_scale_vars : list[str].
        - log1p_vars : list[str].
        - log_vars : list[str].
        - dropfirst_onehot : bool. 
            Default: False. 
            All binary variables will automatically drop first, 
            regardless of the value of  dropfirst_onehot

        Returns
        -------
        - None
        """
        self._dp = DataPreprocessor(
            self._working_df_train,
            imputation_strategy=imputation_strategy,
            onehot_vars=onehot_vars,
            standardize_vars=standardize_vars,
            minmax_vars=minmax_vars,
            log1p_vars=log1p_vars,
            log_vars=log_vars,
            dropfirst_onehot=dropfirst_onehot,
            verbose=self._tm_verbose
        )
        self._working_df_train = self._dp.forward(
            self._working_df_train)
        self._working_df_test = self._dp.forward(
            self._working_df_test)
        self._working_train_test_var_agreement()
        self._reset_categorical_continuous_vars()
        if self._tm_verbose:
            print_wrapped(f'{color_text("UPDATE:", "green")} ' +  \
                'Preprocessing complete. ' +\
                'Re-identified categorical ' +\
                'and continuous variables.')


    def voting_selection(self, X_vars: list[str], y_var: str, 
                         selectors: Iterable[RegressionBaseSelector], 
                         n_target_features: int, 
                         update_working_dfs: bool = False):
        """Supervised feature selection via methods voting based on
        training dataset. 
        Also returns a RegressionVotingSelectionReport object. 
        Can automatically update the working train and working test
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

        Returns
        -------
        - RegressionVotingSelectionReport
        """

        report = RegressionVotingSelectionReport(self._working_df_train, 
            X_vars, y_var, selectors, n_target_features, 
            verbose=self._tm_verbose)
        if update_working_dfs:
            self._working_df_test = self._working_df_test[report.top_features]
            self._working_df_train = self._working_df_train[report.top_features]
            self._reset_categorical_continuous_vars()
        return report
    

    def lm(self, X_vars: list[str], y_var: str, 
           regularization_type: Literal[None, 'l1', 'l2'] = None, 
           alpha: float = 0.0, inverse_scale_y: bool = True):
        """Conducts a simple OLS regression analysis exercise. 

        Parameters
        ----------
        - X_vars : list[str]. 
        - y_var : str. 
        - regularization_type : [None, 'l1', 'l2']. 
            Default: None.
        - alpha : float.
            Default: 0.
        - inverse_scale_y : bool.
            If True, inverse scales the y_true and y_pred values to their 
            original scales. Default: 0.
            
        Returns
        -------
        - train_report : LinearRegressionReport.
        - test_report : LinearRegressionReport.
        """
        local_X_train_df = self._working_df_train[X_vars]
        local_X_test_df = self._working_df_test[X_vars]
        local_y_train_series = self._working_df_train[y_var]
        local_y_test_series = self._working_df_test[y_var]
        model = OrdinaryLeastSquares(X=local_X_train_df, y=local_y_train_series,
                                     regularization_type=regularization_type,
                                     alpha=alpha)
        model.fit()
        y_var_scaler = None
        if self._dp is not None and inverse_scale_y:
            y_var_scaler = self._dp.get_single_var_scaler(y_var)
        return (
            LinearRegressionReport(model, X_eval=local_X_train_df, 
                                   y_eval=local_y_train_series,
                                   y_scaler=y_var_scaler),
            LinearRegressionReport(model, X_eval=local_X_test_df,
                                   y_eval=local_y_test_series,
                                   y_scaler=y_var_scaler),
        )
    

    def lm_rlike(self, formula: str, inverse_scale_y: bool = True):
        """Performs an R-like regression analysis. That is, 
        all further preprocessing should be specified 
        in the formula; any categorical variables are automatically 
        detected and one-hot encoded. 
        Examples with missing data will be dropped.

        Parameters
        ----------
        - formula : str. 
            An R-like formula, e.g. y ~ x1 + log(x2) + poly(x3)
        - inverse_scale_y : bool.
            If True, inverse scales the y_true and y_pred values to their 
            original scales. Default: 0.

        Returns
        -------
        - train_report : LinearRegressionReport.
        - test_report : LinearRegressionReport.
        """

        y_series_train, y_scaler_train, X_df_train =\
            parse_and_transform_rlike(formula, self._working_df_train)
        y_series_test, y_scaler_test, X_df_test =\
            parse_and_transform_rlike(formula, self._working_df_test)
        
        y_var = y_series_train.name
        X_vars = X_df_train.columns

        # ensure missing values are dropped
        y_X_df_combined_train = pd.DataFrame(y_series_train).join(X_df_train)
        y_X_df_combined_test = pd.DataFrame(y_series_test).join(X_df_test)
        y_X_df_combined_train.dropna(inplace=True)
        y_X_df_combined_test.dropna(inplace=True)
        y_X_df_combined_train, y_X_df_combined_test =\
            self._train_test_var_agreement(y_X_df_combined_train, 
                y_X_df_combined_test)

        y_series_train = y_X_df_combined_train[y_var]
        X_df_train = y_X_df_combined_train[X_vars]
        y_series_test = y_X_df_combined_test[y_var]
        X_df_test = y_X_df_combined_test[X_vars]

        model = OrdinaryLeastSquares(X=X_df_train, y=y_series_train)
        model.fit()

        if not inverse_scale_y:
            y_scaler_train = None
            y_scaler_test = None
        return (
            LinearRegressionReport(model, X_eval=X_df_train, 
                                   y_eval=y_series_train,
                                   y_scaler=y_scaler_train),
            LinearRegressionReport(model, X_eval=X_df_test,
                                   y_eval=y_series_test,
                                   y_scaler=y_scaler_test),
        )



    # --------------------------------------------------------------------------
    # MACHINE LEARNING
    # --------------------------------------------------------------------------

    def ml_regression_benchmarking(self, X_vars: list[str], y_var: str, 
                                   models: Iterable[BaseRegression], 
                                   outer_cv: int | None = None,
                                   outer_cv_seed: int = 42, 
                                   inverse_scale_y: bool = True):
        """Conducts a comprehensive regression benchmarking exercise. 

        Parameters
        ----------
        - X_vars : list[str]. 
        - y_var : str. 
        - models : Iterable[BaseRegression]. 
            Testing performance of all models will be evaluated. 
        - outer_cv : int.
            If not None, reports training scores via nested k-fold CV.
        - outer_cv_seed : int.
            The random seed for the outer cross validation loop.
        - inverse_scale_y : bool.
            If true, inverse scales the y_true and y_pred values to their 
            original scales. Default: 0.
        
        Returns
        -------
        - train_report : ComprehensiveMLRegressionReport.
        - test_report : ComprehensiveMLRegressionReport.
        """
        local_X_train_df = self._working_df_train[X_vars]
        local_X_test_df = self._working_df_test[X_vars]
        local_y_train_series = self._working_df_train[y_var]
        local_y_test_series = self._working_df_test[y_var]
        X_train_np = local_X_train_df.to_numpy()
        y_train_np = local_y_train_series.to_numpy().flatten()
        for i, model in enumerate(models):
            if self._tm_verbose:
                print_wrapped(f'{color_text("UPDATE:", "green")} ' \
                    + f'Task {i+1} of ' +\
                    f'{len(models)}.\tFitting {model}.')
            model.fit(X_train_np, y_train_np, outer_cv=outer_cv, 
                      outer_cv_seed=outer_cv_seed)
        y_var_scaler = None
        if self._dp is not None and inverse_scale_y:
            y_var_scaler = self._dp.get_single_var_scaler(y_var)
        train_report = ComprehensiveMLRegressionReport(
            models=models, y_scaler=y_var_scaler)
        test_report = ComprehensiveMLRegressionReport(
            models, local_X_test_df, local_y_test_series, y_var_scaler)
        return train_report, test_report




    # --------------------------------------------------------------------------
    # DATAFRAME MANIPULATION + INDEXING
    # --------------------------------------------------------------------------
    
    def save_data_checkpoint(self, checkpoint: str):
        """Saves the current state of the working train and test datasets. 
        The state may be returned to by calling reset_working_dfs(checkpoint).

        Parameters
        ----------
        - checkpoint : str. 

        Returns
        -------
        - None
        """
        if self._tm_verbose:
            print_wrapped(f'{color_text("UPDATE:", "green")} ' + \
                'Working datasets ' + \
                f'checkpoint "{checkpoint}" saved.')
        self._df_checkpoint_name_to_df[checkpoint] = (
            self._working_df_train.copy(),
            self._working_df_test.copy()
        )
        
    def load_data_checkpoint(self, checkpoint: str = None):
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
            if self._tm_verbose:
                shapes_dict = self.shapes()
                print_wrapped(f'{color_text("UPDATE:", "green")} ' + \
                      'Working datasets reset to original datasets. ' +\
                      'Shapes of train, test datasets: ' + \
                    f'{shapes_dict["train"]}, {shapes_dict["test"]}')
            self._working_df_test = self._original_df_test.copy()
            self._working_df_train = self._original_df_train.copy()
        else:
            if self._tm_verbose:
                shapes_dict = self.shapes()
                print_wrapped(f'{color_text("UPDATE:", "green")} ' + \
                       'Working datasets reset to checkpoint ' +\
                       f'"{checkpoint}". ' +\
                        'Shapes of train, test datasets: ' + \
                    f'{shapes_dict["train"]}, {shapes_dict["test"]}')
            self._working_df_train =\
                self._df_checkpoint_name_to_df[checkpoint][0].copy()
            self._working_df_test =\
                self._df_checkpoint_name_to_df[checkpoint][1].copy()
        self._reset_categorical_continuous_vars()


    def remove_data_checkpoint(self, checkpoint: str):
        """Removes a saved checkpoint to conserve memory.

        Parameters
        ----------
        - checkpoint : str. 

        Returns
        -------
        - None
        """
        out_chkpt = self._df_checkpoint_name_to_df.pop(checkpoint)
        if self._tm_verbose:
            print_wrapped(f'{color_text("UPDATE:", "green")} Removed working '+\
                  f'dataset checkpoint {out_chkpt}.')
    

    def select_vars(self, vars: list[str]):
        """Selects subset of (column) variables in-place on the working 
        train and test datasets. 

        Parameters
        ----------
        - vars : list[str]
        """
        self._working_df_train = self._working_df_train[vars]
        self._working_df_test = self._working_df_test[vars]
        self._reset_categorical_continuous_vars()
        if self._tm_verbose:
            shapes_dict = self.shapes()
            print_wrapped(f'{color_text("UPDATE:", "green")} ' + \
                  f'Selected columns {vars}. ' +\
                  'Re-identified categorical ' +\
                  'and continuous variables. ' +\
                    'Shapes of train, test datasets: ' + \
                    f'{shapes_dict["train"]}, {shapes_dict["test"]}')


    def drop_vars(self, vars: list[str]):
        """Drops subset of variables (columns) in-place on the working 
        train and test datasets. 

        Parameters
        ----------
        - vars : list[str]
        """
        self._working_df_train.drop(vars, axis='columns', inplace=True)
        self._working_df_test.drop(vars, axis='columns', inplace=True)
        self._reset_categorical_continuous_vars()
        if self._tm_verbose:
            shapes_dict = self.shapes()
            print_wrapped(f'{color_text("UPDATE:", "green")} ' +\
                f'Dropped columns {vars}. '+\
                'Re-identified categorical ' +\
                'and continuous variables. ' +\
                'Shapes of train, test datasets: ' + \
                f'{shapes_dict["train"]}, {shapes_dict["test"]}')


    def drop_train_examples(self, indices: list):
        """Drops subset of examples (rows) in-place on the working train 
        dataset. 

        Parameters
        ----------
        - indices : list.
        """
        self._working_df_train.drop(indices, axis='index', inplace=True)
        if self._tm_verbose:
            shapes_dict = self.shapes()
            print_wrapped(f'{color_text("UPDATE:", "green")} ' +\
                f'Dropped rows {indices}. '+\
                'Shapes of train, test datasets: ' + \
                f'{shapes_dict["train"]}, {shapes_dict["test"]}')


    # --------------------------------------------------------------------------
    # GETTERS
    # --------------------------------------------------------------------------
    def shapes(self):
        """Returns a dictionary containing shape information for the 
        TabularMagic (working) datasets
        
        Returns
        -------
        - {
            'train': self.working_df_train.shape,
            'test': self.working_df_test.shape
        }
        """
        return {
            'train': self._working_df_train.shape,
            'test': self._working_df_test.shape
        }


    def dataset(self, dataset: Literal['train', 'test'] = 'train', 
                copy: bool = False):
        """Returns one of (working_df_train, working_df_test). 

        Parameters
        ----------
        - dataset : Literal['train', 'test']
        - copy : bool. If True, dataframe is copied before being returned. 
        
        Returns
        -------
        - df : pd.DataFrame
        """
        if dataset == 'train':
            if copy:
                return self._working_df_train.copy()
            else: 
                return self._working_df_train
            
        elif dataset == 'test':
            if copy:
                return self._working_df_test.copy()
            else:
                return self._working_df_test
        
        else:
            raise ValueError('Invalid input for dataset.')
    

    def head(self, n = 5):
        """Same as calling self.working_df_train.head(n)."""
        return self._working_df_train.head(n)
    

    def continuous_vars(self, copy: bool = False):
        """Returns copy of list of continuous variables."""
        if copy:
            return self._continuous_vars.copy()
        else:
            return self._continuous_vars

    def categorical_vars(self, copy: bool = False):
        """Returns copy of list of categorical variables."""
        if copy:
            return self._categorical_vars.copy()
        else:
            return self._categorical_vars


    def __len__(self):
        """Returns the number of examples in working_df_train.
        """
        return len(self._working_df_train)



    # --------------------------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------------------------
    def _verify_input_dfs(self):
        """Ensures that the original train and test datasets have the 
        same variables. 
        """
        l1 = set(self._original_df_test.columns.to_list())
        l2 = set(self._original_df_train.columns.to_list())
        if len(l1.union(l2)) != len(l1):
            raise RuntimeWarning('The train dataset and test dataset' + \
                ' do not have the same variables.')


    def _reset_categorical_continuous_vars(self):
        """Resets the categorical and continuous column values."""
        self._categorical_vars = self._working_df_train.select_dtypes(
            include=['object', 'category', 'bool']).columns.to_list()
        self._continuous_vars = self._working_df_train.select_dtypes(
            exclude=['object', 'category', 'bool']).columns.to_list()
        if self._continuous_vars is None:
            self._continuous_vars = []
        if self._categorical_vars is None:
            self._categorical_vars = []
        

    def _working_train_test_var_agreement(self):
        missing_test_columns = list(set(self._working_df_train.columns) -\
            set(self._working_df_test.columns))
        extra_test_columns = list(set(self._working_df_test.columns) -\
            set(self._working_df_train.columns))
        if len(extra_test_columns) > 0:
            self._working_df_test.drop(columns=extra_test_columns, axis=1, 
                                      inplace=True)
        if len(missing_test_columns) > 0:
            for col in missing_test_columns:
                self._working_df_test[col] = 0
        assert len(self._working_df_test.columns) == \
            len(self._working_df_train.columns)

        # ensure that the train and test dfs have the same order
        # (for nicer exploration)
        self._working_df_test = self._working_df_test[\
            self._working_df_train.columns]

        for a, b in zip(self._working_df_test.columns, 
                        self._working_df_train.columns):
            assert a == b


    def _train_test_var_agreement(self, df_train: pd.DataFrame, 
                                  df_test: pd.DataFrame):
        """Modifies df_test to have the same columns as df_train. This helps 
        mitigate any problems that may arise from one-hot-encoding the test set.

        Parameters
        ----------
        - df_train : pd.DataFrame.
        - df_test : pd.DataFrane.

        Returns
        -------
        - df_train : pd.DataFrame
        - df_test : pd.DataFrame
        """
        missing_test_columns = list(set(df_train.columns) -\
            set(df_test.columns))
        extra_test_columns = list(set(df_test.columns) -\
            set(df_train.columns))
        if len(extra_test_columns) > 0:
            print_wrapped(f'{color_text("WARNING:", "red")} ' +\
                  f'Columns {extra_test_columns} not found in train ' +\
                   'have been dropped from test')
            self._working_df_test.drop(columns=extra_test_columns, axis=1, 
                                      inplace=True)
        if len(missing_test_columns) > 0:
            print_wrapped(f'{color_text("WARNING:", "red")} ' +\
                  f'Columns {missing_test_columns} not found in test ' +\
                   'have been added to test with 0-valued entries')
            for col in missing_test_columns:
                df_test[col] = 0
        assert len(df_test.columns) == \
            len(df_train.columns)

        # ensure that the train and test dfs have the same order
        # (for nicer exploration)
        df_test = df_test[df_train.columns]
        for a, b in zip(df_test.columns, 
                        df_train.columns):
            assert a == b
        return df_train, df_test


    def _remove_spaces_varnames(self):
        """Removes spaces from variable names. Necessary for R-like lm()
        calls.
        """
        new_columns = self._working_df_train.columns.to_list()
        for i, var in enumerate(new_columns):
            new_columns[i] = ''.join(var.split(' '))
        self._working_df_train.columns = new_columns
        self._working_df_test.columns = new_columns

            

    def __str__(self):
        """Returns metadata in string form. """


        max_width = TOSTR_MAX_WIDTH


        textlen_shapes = len(str(self._working_df_train.shape) +\
            str(self._working_df_test.shape)) + 25
        shapes_message_buffer_left = (max_width - textlen_shapes) // 2
        shapes_message_buffer_right = math.ceil(
            (max_width - textlen_shapes) / 2)


        shapes_message = color_text('Train shape: ', 'none') +\
            f'{self._working_df_train.shape}'+ \
            ' '*shapes_message_buffer_left +\
            color_text('Test shape: ', 'none') + \
            f'{self._working_df_test.shape}'  + \
            ' '*shapes_message_buffer_right


        title_message = color_text(self._id, 'none')
        title_message = fill(title_message, width=max_width)
        
        categorical_var_message = color_text('Categorical variables:', 'none')
        if len(self._categorical_vars) == 0:
            categorical_var_message += color_text(
                ' None', 'purple')
        for i, var in enumerate(self._categorical_vars):
            categorical_var_message += f' {var}'
            if i < len(self._categorical_vars) - 1:
                categorical_var_message += ','
        categorical_var_message = fill(categorical_var_message, 
                                       drop_whitespace=False, width=max_width)

        continuous_var_message = color_text('Continuous variables:', 'none')
        if len(self._continuous_vars) == 0:
            continuous_var_message += color_text(
                ' None', 'purple')
        for i, var in enumerate(self._continuous_vars):
            continuous_var_message += f' {var}'
            if i < len(self._continuous_vars) - 1:
                continuous_var_message += ','
        continuous_var_message = fill(continuous_var_message, width=max_width)

        bottom_divider = '\n' + color_text('='*max_width, 'none')
        divider = '\n' + color_text('-'*max_width, 'none') + '\n'
        divider_invisible = '\n' + ' '*max_width + '\n'
        top_divider = color_text('='*max_width, 'none') + '\n'

        final_message = top_divider + title_message + divider +\
            shapes_message + divider + categorical_var_message +\
            divider_invisible + continuous_var_message + bottom_divider
        
        return final_message


        




