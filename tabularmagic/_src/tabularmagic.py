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
from .interactive import (MLRegressionReport, 
    ComprehensiveEDA, RegressionVotingSelectionReport, 
    LinearRegressionReport)
from .util.console import print_wrapped, color_text
from .util.constants import TOSTR_MAX_WIDTH
from .feature_selection import RegressionBaseSelector
from .data.datahandler import DataHandler



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
            Default: None. If not None, then treats df as the train dataset.
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

        self._verbose = verbose


        if df_test is not None:
            self._datahandler = DataHandler(
                df_train=df,
                df_test=df_test,
                verbose=self._verbose
            )

        else:
            if test_size > 0:
                temp_train, temp_test = train_test_split(df, 
                    test_size=test_size, shuffle=True, 
                    random_state=split_seed)
                temp_train_df = pd.DataFrame(temp_train, columns=df.columns)
                temp_test_df = pd.DataFrame(temp_test, columns=df.columns)
            else:
                if self._verbose:
                    print_wrapped(
                        'No test dataset provided. ' +\
                        'Test dataset will be treated as train dataset copy.',
                        type='WARNING'
                    )
                temp_train_df = df
                temp_test_df = df
            self._datahandler = DataHandler(
                df_train=temp_train_df,
                df_test=temp_test_df, 
                verbose=self._verbose
            )
        self._id = id

        if self._verbose:
            shapes_dict = self._datahandler.shapes()
            print_wrapped(
                'TabularMagic initialization complete. ' +\
                'Shapes of train, test datasets: ' + \
                f'{shapes_dict["train"]}, {shapes_dict["test"]}.',
                type='UPDATE'
            )


    # --------------------------------------------------------------------------
    # EDA + FEATURE SELECTION + REGRESSION ANALYSIS
    # --------------------------------------------------------------------------
    def eda(self, dataset: Literal['train', 'test'] = 'train') ->\
            ComprehensiveEDA:
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
            return ComprehensiveEDA(self._datahandler.df_train())
        elif dataset == 'test':
            return ComprehensiveEDA(self._datahandler.df_test())
        else:
            raise ValueError(f'Invalid input: dataset = {dataset}.')


    def feature_selection(self, selectors: Iterable[RegressionBaseSelector], 
                         y_var: str, 
                         X_vars: list[str] = None, 
                         n_target_features: int = 10, 
                         update_working_dfs: bool = False):
        """Supervised feature selection via methods voting based on
        training dataset. 
        Also returns a RegressionVotingSelectionReport object. 
        Can automatically update the working train and working test
        datasets so that only the selected features remain if 
        update_working_dfs is True.
        
        Parameters
        ----------
        - selectors : Iterable[BaseSelector].
            Each BaseSelector decides on the top n_target_features.
        - y_var : str.
            The variable to be predicted. 
        - X_vars : list[str].
            A list of features from which n_target_features are to be selected.
            If None, all continuous variables except y_var will be used. 
        - n_target_features : int. 
            Number of desired features, < len(X_vars). Default 10. 
        - update_working_dfs : bool.
            Default: False.

        Returns
        -------
        - RegressionVotingSelectionReport
        """
        if X_vars is None:
            X_vars = self._datahandler.continuous_vars(True)
            X_vars.remove(y_var)
        report = RegressionVotingSelectionReport(self._datahandler.df_train(),
            X_vars, y_var, selectors, n_target_features, 
            verbose=self._verbose)
        if update_working_dfs:
            var_subset = report._top_features + [y_var]
            self._datahandler.select_vars(var_subset)
        return report
    


    def lm(self, y_var: str, X_vars: list[str] = None, 
           regularization_type: Literal[None, 'l1', 'l2'] = None, 
           alpha: float = 0.0, inverse_scale_y: bool = True) ->\
                LinearRegressionReport:
        """Conducts a simple OLS regression analysis exercise. 

        Parameters
        ----------
        - y_var : str. 
        - X_vars : list[str]. 
            If None, all continuous variables except y_var will be used. 
        - regularization_type : [None, 'l1', 'l2']. 
            Default: None.
        - alpha : float.
            Default: 0.
        - inverse_scale_y : bool.
            If True, inverse scales the y_true and y_pred values to their 
            original scales. Default: 0.
            
        Returns
        -------
        - report : LinearRegressionReport
        """
        if X_vars is None:
            X_vars = self._datahandler.continuous_vars(True)
            X_vars.remove(y_var)
        y_scaler = None
        if inverse_scale_y:
            y_scaler = self._datahandler.yscaler()
        return LinearRegressionReport(
            OrdinaryLeastSquares(
                regularization_type=regularization_type,
                alpha=alpha), 
            self._datahandler, X_vars, y_var, y_scaler
        )



    def lm_rlike(self, formula: str, inverse_scale_y: bool = True) -> \
            LinearRegressionReport:
        """Performs an R-like regression with OLS. That is, 
        all further preprocessing should be specified in the formula; 
        any categorical variables are automatically detected and 
        one-hot encoded. Examples with missing data will be dropped. 

        Parameters
        ----------
        - formula : str. 
            An R-like formula, e.g. y ~ x1 + log(x2) + poly(x3) + x1 * x2
        - inverse_scale_y : bool.
            If True, inverse scales the y_true and y_pred values to their 
            original scales. Default: 0.

        Returns
        -------
        - report : LinearRegressionReport
        """

        try:
            y_series_train, y_scaler, X_df_train =\
                parse_and_transform_rlike(formula, self._datahandler.df_train())
            y_series_test, _, X_df_test =\
                parse_and_transform_rlike(formula, self._datahandler.df_test())
        except:
            raise ValueError(f'Invalid formula: {formula}')
        
        y_var = y_series_train.name
        X_vars = X_df_train.columns

        # ensure missing values are dropped
        y_X_df_combined_train = pd.DataFrame(y_series_train).join(X_df_train)
        y_X_df_combined_test = pd.DataFrame(y_series_test).join(X_df_test)
        y_X_df_combined_train.dropna(inplace=True)
        y_X_df_combined_test.dropna(inplace=True)
        y_X_df_combined_train, y_X_df_combined_test =\
            self._datahandler._force_train_test_var_agreement(
                y_X_df_combined_train, 
                y_X_df_combined_test)
        if not inverse_scale_y:
            y_scaler = None

        return LinearRegressionReport(
            OrdinaryLeastSquares(), 
            DataHandler(y_X_df_combined_train, y_X_df_combined_test, False), 
            X_vars, y_var, y_scaler
        )



    # --------------------------------------------------------------------------
    # MACHINE LEARNING
    # --------------------------------------------------------------------------

    def ml_regression(self, models: Iterable[BaseRegression], 
                    y_var: str, X_vars: list[str] = None,
                    outer_cv: int = None,
                    outer_cv_seed: int = 42, 
                    inverse_scale_y: bool = True) ->\
                        MLRegressionReport:
        """Conducts a comprehensive regression benchmarking exercise. 

        Parameters
        ----------
        - models : Iterable[BaseRegression]. 
            Testing performance of all models will be evaluated. 
        - y_var : str. 
        - X_vars : list[str]. 
            If None, uses all continuous variables except y_var as predictors.
        - outer_cv : int.
            If not None, reports training scores via nested k-fold CV.
        - outer_cv_seed : int.
            The random seed for the outer cross validation loop.
        - inverse_scale_y : bool.
            If true, inverse scales the y_true and y_pred values to their 
            original scales. Default: 0.
        
        Returns
        -------
        - report : ComprehensiveMLRegressionReport
        """
        if X_vars is None:
            X_vars = self._datahandler.continuous_vars(True)
            X_vars.remove(y_var)
        y_scaler = None
        if self._dp is not None and inverse_scale_y:
            y_scaler = self._dp.get_single_var_scaler(y_var)
        return MLRegressionReport(
            models=models,
            datahandler=self._datahandler,
            X_vars=X_vars,
            y_var=y_var,
            y_scaler=y_scaler,
            outer_cv=outer_cv,
            outer_cv_seed=outer_cv_seed,
            verbose=self._verbose
        )




    # --------------------------------------------------------------------------
    # DATAFRAME MANIPULATION + INDEXING
    # --------------------------------------------------------------------------
    
    def save_data_checkpoint(self, checkpoint: str):
        """Saves the current state of the working train and test datasets. 
        The state may be returned to by calling reset_working_dfs(checkpoint).

        Parameters
        ----------
        - checkpoint : str. 
        """
        self._datahandler.save_data_checkpoint(checkpoint)
        
    def load_data_checkpoint(self, checkpoint: str = None):
        """The working train and working test datasets are reset to the 
        input datasets given at object initialization.

        Parameters
        ----------
        - checkpoint : str. 
            Default: None. If None, sets the working datasets to the original 
            datasets given at object initialization. 
        """
        self._datahandler.load_data_checkpoint(checkpoint)

    def remove_data_checkpoint(self, checkpoint: str):
        """Removes a saved checkpoint to conserve memory.

        Parameters
        ----------
        - checkpoint : str. 
        """
        self._datahandler.remove_data_checkpoint(checkpoint)
        

    def select_vars(self, vars: list[str]):
        """Selects subset of (column) variables in-place on the working 
        train and test datasets. 

        Parameters
        ----------
        - vars : list[str]
        """
        self._datahandler.select_vars(vars)


    def drop_vars(self, vars: list[str]):
        """Drops subset of variables (columns) in-place on the working 
        train and test datasets. 

        Parameters
        ----------
        - vars : list[str]
        """
        self._datahandler.drop_vars(vars)


    def drop_train_examples(self, indices: list):
        """Drops subset of examples (rows) in-place on the working train 
        dataset. 

        Parameters
        ----------
        - indices : list.
        """
        self._datahandler.drop_train_examples(indices)


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
        return self._datahandler.shapes()
    
    
    def datahandler(self) -> DataHandler:
        """Returns the DataHandler."""
        return self._datahandler


    def __len__(self):
        """Returns the number of examples in working train DataFrame."""
        return len(self._datahandler)
            

    def __str__(self):
        """Returns metadata in string form."""
        working_df_test = self._datahandler.df_test()
        working_df_train = self._datahandler.df_train()

        max_width = TOSTR_MAX_WIDTH

        textlen_shapes = len(str(working_df_train.shape) +\
            str(working_df_test.shape)) + 25
        shapes_message_buffer_left = (max_width - textlen_shapes) // 2
        shapes_message_buffer_right = math.ceil(
            (max_width - textlen_shapes) / 2)


        shapes_message = color_text('Train shape: ', 'none') +\
            f'{working_df_train.shape}'+ \
            ' '*shapes_message_buffer_left +\
            color_text('Test shape: ', 'none') + \
            f'{working_df_test.shape}'  + \
            ' '*shapes_message_buffer_right


        title_message = color_text(self._id, 'none')
        title_message = fill(title_message, width=max_width)
        
        categorical_var_message = color_text('Categorical variables:', 'none')
        if len(self._datahandler.categorical_vars()) == 0:
            categorical_var_message += color_text(
                ' None', 'purple')
        for i, var in enumerate(self._datahandler.categorical_vars()):
            categorical_var_message += f' {var}'
            if i < len(self._datahandler.categorical_vars()) - 1:
                categorical_var_message += ','
        categorical_var_message = fill(categorical_var_message, 
                                       drop_whitespace=False, width=max_width)

        continuous_var_message = color_text('Continuous variables:', 'none')
        if len(self._datahandler.continuous_vars()) == 0:
            continuous_var_message += color_text(
                ' None', 'purple')
        for i, var in enumerate(self._datahandler.continuous_vars()):
            continuous_var_message += f' {var}'
            if i < len(self._datahandler.continuous_vars()) - 1:
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


        




    def _repr_pretty_(self, p, cycle):
        p.text(str(self))


        