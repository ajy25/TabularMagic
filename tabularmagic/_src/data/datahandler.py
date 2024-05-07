import pandas as pd
from typing import Literal
from ..util.console import print_wrapped
from .preprocessing import (BaseSingleVarScaler, Log1PTransformSingleVar, 
    LogTransformSingleVar, MinMaxSingleVar, StandardizeSingleVar)




class DataHandler:
    """DataHandler: handles all aspects of data preprocessing and loading.
    """

    def __init__(self, df_train: pd.DataFrame, 
                 df_test: pd.DataFrame, 
                 verbose: bool = True):
        """Initializes a DataHandler object.

        Parameters
        ----------
        - df_train : pd.DataFrame.
        - df_test : pd.DataFrame.
        - verbose : bool.
        """
        self._checkpoint_name_to_df: \
            dict[str, tuple[pd.DataFrame, pd.DataFrame]] = dict()
        self._verbose = verbose

        # verify and set the original DataFrames
        self._verify_input_dfs(df_train, df_test)
        df_train, df_test = self._remove_spaces_varnames(df_train, df_test)
        self._orig_df_train, self._orig_df_test =\
            self._force_train_test_var_agreement(df_train.copy(), 
                                                 df_test.copy())
        self._categorical_vars, self._continuous_vars =\
            self._compute_categorical_continuous_vars(self._orig_df_train)

        # set the working DataFrames
        self._working_df_train = self._orig_df_train.copy()
        self._working_df_test = self._orig_df_test.copy()
    

    # --------------------------------------------------------------------------
    # CHECKPOINT HANDLING
    # --------------------------------------------------------------------------
    def load_data_checkpoint(self, checkpoint: str = None):
        """The working train and working test datasets are reset to the 
        input datasets given at object initialization.

        Parameters
        ----------
        - checkpoint : str. 
            Default: None. If None, sets the working datasets to the original 
            datasets given at object initialization. 
        """
        if checkpoint is None:
            if self._verbose:
                shapes_dict = self.shapes()
                print_wrapped(
                    'Working datasets reset to original datasets. ' +\
                    'Shapes of train, test datasets: ' + \
                    f'{shapes_dict["train"]}, {shapes_dict["test"]}.',
                    type='UPDATE'
                )
            self._working_df_test = self._orig_df_test.copy()
            self._working_df_train = self._orig_df_train.copy()
        else:
            if self._verbose:
                shapes_dict = self.shapes()
                print_wrapped(
                    'Working datasets reset to checkpoint ' +\
                    f'"{checkpoint}". ' +\
                    'Shapes of train, test datasets: ' + \
                    f'{shapes_dict["train"]}, {shapes_dict["test"]}.', 
                    type='UPDATE'
                )
            self._working_df_train =\
                self._checkpoint_name_to_df[checkpoint][0].copy()
            self._working_df_test =\
                self._checkpoint_name_to_df[checkpoint][1].copy()
        self._categorical_vars, self._continuous_vars =\
            self._compute_categorical_continuous_vars(self._working_df_train)


    def save_data_checkpoint(self, checkpoint: str):
        """Saves the current state of the working train and test datasets. 
        The state may be returned to by calling 
        load_data_checkpoint(checkpoint).

        Parameters
        ----------
        - checkpoint : str. 
        """
        if self._verbose:
            print_wrapped(
                f'Saved working datasets checkpoint "{checkpoint}".',
                type='UPDATE'
            )
        self._checkpoint_name_to_df[checkpoint] = (
            self._working_df_train.copy(),
            self._working_df_test.copy()
        )


    def remove_data_checkpoint(self, checkpoint: str):
        """Removes a saved checkpoint to conserve memory.

        Parameters
        ----------
        - checkpoint : str. 
        """
        out_chkpt = self._checkpoint_name_to_df.pop(checkpoint)
        if self._verbose:
            print_wrapped(
                f'Removed working dataset checkpoint "{out_chkpt}".',
                type='UPDATE'
            )


    # --------------------------------------------------------------------------
    # DATAFRAME MANIPULATION + INDEXING
    # --------------------------------------------------------------------------
    def select_vars(self, vars: list[str]):
        """Selects subset of (column) variables in-place on the working 
        train and test datasets. 

        Parameters
        ----------
        - vars : list[str]
        """
        self._working_df_train = self._working_df_train[vars]
        self._working_df_test = self._working_df_test[vars]
        self._categorical_vars, self._continuous_vars =\
            self._compute_categorical_continuous_vars()
        if self._verbose:
            shapes_dict = self.shapes()
            print_wrapped(
                f'Selected columns {vars}. ' +\
                'Re-identified categorical ' +\
                'and continuous variables. ' +\
                'Shapes of train, test datasets: ' + \
                f'{shapes_dict["train"]}, {shapes_dict["test"]}.',
                type='UPDATE'
            )


    def drop_vars(self, vars: list[str]):
        """Drops subset of variables (columns) in-place on the working 
        train and test datasets. 

        Parameters
        ----------
        - vars : list[str]
        """
        self._working_df_train.drop(vars, axis='columns', inplace=True)
        self._working_df_test.drop(vars, axis='columns', inplace=True)
        self._categorical_vars, self._continuous_vars =\
            self._compute_categorical_continuous_vars(self._working_df_train)
        if self._verbose:
            shapes_dict = self.shapes()
            print_wrapped(
                f'Dropped columns {vars}. '+\
                'Re-identified categorical ' +\
                'and continuous variables. ' +\
                'Shapes of train, test datasets: ' + \
                f'{shapes_dict["train"]}, {shapes_dict["test"]}.', 
                type='UPDATE'
            )

    def drop_train_examples(self, indices: list):
        """Drops subset of examples (rows) in-place on the working train 
        dataset. 

        Parameters
        ----------
        - indices : list.
        """
        self._working_df_train.drop(indices, axis='index', inplace=True)
        if self._verbose:
            shapes_dict = self.shapes()
            print_wrapped(
                f'Dropped rows {indices}. '+\
                'Shapes of train, test datasets: ' + \
                f'{shapes_dict["train"]}, {shapes_dict["test"]}.',
                type='UPDATE'
            )


    # --------------------------------------------------------------------------
    # GETTERS
    # --------------------------------------------------------------------------
    def df_train(self, copy: bool = False):
        """Returns the working train DataFrame.

        Parameters
        ----------
        - copy : bool.
            If True returns a copy.

        Returns
        -------
        - pd.DataFrame
        """
        if copy:
            return self._working_df_train.copy()
        return self._working_df_train


    def df_test(self, copy: bool = False):
        """Returns the working test DataFrame.

        Parameters
        ----------
        - copy : bool.
            If True returns a copy.

        Returns
        -------
        - pd.DataFrame
        """
        if copy:
            return self._working_df_test.copy()
        return self._working_df_test
    

    def shapes(self):
        """Returns a dictionary containing shape information for the 
        working DataFrames.
        
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
    

    def force_categorical(self, vars: list[str]):
        """Forces variables to become categorical. 
        Example use case: prepare numerically-coded categorical variables 

        Parameters
        ----------
        - vars : list[str]
        """
        self._working_df_train[vars] =\
            self._working_df_train[vars].astype('str')
        if self._verbose:
            print_wrapped(
                f'Converted variables {vars} to categorical.',
                type='UPDATE'
            )
        self._categorical_vars, self._continuous_vars =\
            self._compute_categorical_continuous_vars(self._working_df_train)


    # --------------------------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------------------------


    def onehot(self, inplace: bool = True):
        """One-hot encodes the train and test DataFrames."""
        
    
    def handle_missing(self, strategy: Literal['dropna'], 
                       included_vars: list[str] = None,
                       excluded_vars: list[str] = None, 
                       inplace: bool = True):
        """Handles missing data."""





    # --------------------------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------------------------
    def _verify_input_dfs(self, df1: pd.DataFrame, df2: pd.DataFrame):
        """Ensures that the original train and test datasets have the 
        same variables. 

        Parameters
        ----------
        - df1 : pd.DataFrame
        - df2 : pd.DataFrame
        """
        l1 = set(df1.columns.to_list())
        l2 = set(df2.columns.to_list())
        if len(l1.union(l2)) != len(l1):
            raise RuntimeWarning('The train dataset and test dataset' + \
                ' do not have the same variables.')


    def _compute_categorical_continuous_vars(self, df: pd.DataFrame):
        """Resets the categorical and continuous column values.
        
        Parameters
        ----------
        - df : pd.DataFrame

        Returns
        -------
        - categorical_vars : list[str]
        - continuous_vars : list[str]
        """
        categorical_vars = df.select_dtypes(
            include=['object', 'category', 'bool']).columns.to_list()
        continuous_vars = df.select_dtypes(
            exclude=['object', 'category', 'bool']).columns.to_list()
        return categorical_vars, continuous_vars
        

    def _force_train_test_var_agreement(self, df_train: pd.DataFrame, 
            df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            if self._verbose:
                print_wrapped(
                    f'Columns {extra_test_columns} not found in train ' +\
                    'have been dropped from test.',
                    type='WARNING'
                )
            self._working_df_test.drop(columns=extra_test_columns, axis=1, 
                                      inplace=True)
        if len(missing_test_columns) > 0:
            if self._verbose:
                print_wrapped(
                    f'Columns {missing_test_columns} not found in test ' +\
                    'have been added to test with 0-valued entries.',
                    type='WARNING'
                )
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


    def _remove_spaces_varnames(self, df_train: pd.DataFrame, 
                                df_test: pd.DataFrame):
        """Removes spaces from variable names. Necessary for R-like lm()
        calls.

        Parameters
        ----------
        - df_train : pd.DataFrame
        - df_test : pd.DataFrame

        Returns
        -------
        - df_train : pd.DataFrame
        - df_test : pd.DataFrame
        """
        new_columns = df_train.columns.to_list()
        for i, var in enumerate(new_columns):
            new_columns[i] = ''.join(var.split(' '))
        df_train.columns = new_columns
        df_test.columns = new_columns
        return df_train, df_test



    def __len__(self):
        """Returns the number of examples in working_df_train."""
        return len(self._working_df_train)



