import pandas as pd
import numpy as np
from typing import Literal
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.utils._testing import ignore_warnings
from ..util.console import (print_wrapped, color_text, bold_text, 
                            list_to_string, 
                            fill_ignore_format)
from .preprocessing import (BaseSingleVarScaler, Log1PTransformSingleVar, 
    LogTransformSingleVar, MinMaxSingleVar, StandardizeSingleVar)
from ..util.constants import TOSTR_MAX_WIDTH


class DataHandler:
    """DataHandler: handles all aspects of data preprocessing and loading.
    """

    def __init__(self, 
                 df_train: pd.DataFrame, 
                 df_test: pd.DataFrame, 
                 y_var: str | None = None,
                 X_vars: list[str] | None = None,
                 name: str = None,
                 verbose: bool = True):
        """Initializes a DataHandler object.

        Parameters
        ----------
        - df_train : pd.DataFrame.
        - df_test : pd.DataFrame.
        - y_var : str | None. Default: None.
        - X_vars : list[str] | None. Default: None.
        - name : str. Default: None.
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
        
        self._categorical_vars, self._continuous_vars, \
            self._categorical_to_categories =\
            self._compute_categorical_continuous_vars(self._orig_df_train)

        # set the working DataFrames
        self._working_df_train = self._orig_df_train.copy()
        self._working_df_test = self._orig_df_test.copy()

        # keep track of scalers
        self._continuous_var_to_scaler = {
            var: None for var in self._continuous_vars
        }

        # set the y-variable
        self._yvar = y_var
        self._Xvars = X_vars
        if X_vars is None:
            self._Xvars = self.categorical_vars(ignore_yvar=True) +\
                self.continuous_vars(ignore_yvar=True)
        if self._yvar is not None:
            if self._yvar in self._Xvars:
                self._Xvars.remove(self._yvar)


        # set the name
        if name is None:
            self._name = 'DataHandler'
        else:
            self._name = name


    # --------------------------------------------------------------------------
    # CHECKPOINT HANDLING
    # --------------------------------------------------------------------------
    def load_data_checkpoint(self, checkpoint: str = None):
        """The working train and working test DataFrames are reset to the 
        original input DataFrames given at object initialization.

        Parameters
        ----------
        - checkpoint : str. 
            Default: None. If None, sets the working DataFrames to the original 
            DataFrames given at object initialization. 

        Returns
        -------
        - self : DataHandler
        """
        if checkpoint is None:
            self._working_df_test = self._orig_df_test.copy()
            self._working_df_train = self._orig_df_train.copy()
            self._continuous_var_to_scaler = {
                var: None for var in self._continuous_vars
            }
            if self._verbose:
                shapes_dict = self._shapes_str_formatted()
                print_wrapped(
                    'Working DataFrames reset to original DataFrames. ' +\
                    'Shapes of train, test DataFrames: ' + \
                    f'{shapes_dict["train"]}, {shapes_dict["test"]}.',
                    type='UPDATE'
                )
        else:
            self._working_df_test =\
                self._checkpoint_name_to_df[checkpoint][0].copy()
            self._working_df_train =\
                self._checkpoint_name_to_df[checkpoint][1].copy()
            self._continuous_var_to_scaler =\
                self._checkpoint_name_to_df[checkpoint][2].copy()
            if self._verbose:
                shapes_dict = self._shapes_str_formatted()
                print_wrapped(
                    'Working DataFrames reset to checkpoint ' +\
                    f'"{checkpoint}". ' +\
                    'Shapes of train, test DataFrames: ' + \
                    f'{shapes_dict["train"]}, {shapes_dict["test"]}.', 
                    type='UPDATE'
                )
        self._categorical_vars, self._continuous_vars, \
            self._categorical_to_categories =\
            self._compute_categorical_continuous_vars(self._working_df_train)
        
        return self


    def save_data_checkpoint(self, checkpoint: str):
        """Saves the current state of the working train and test DataFrames. 
        The state may be returned to by calling 
        load_data_checkpoint(checkpoint).

        Parameters
        ----------
        - checkpoint : str. 

        Returns
        -------
        - self : DataHandler
        """
        if self._verbose:
            print_wrapped(
                f'Saved working data checkpoint "{checkpoint}".',
                type='UPDATE'
            )
        self._checkpoint_name_to_df[checkpoint] = (
            self._working_df_test.copy(),
            self._working_df_train.copy(),
            self._continuous_var_to_scaler.copy()
        )

        return self


    def remove_data_checkpoint(self, checkpoint: str):
        """Removes a saved checkpoint to conserve memory.

        Parameters
        ----------
        - checkpoint : str. 

        Returns
        -------
        - self : DataHandler
        """
        out_chkpt = self._checkpoint_name_to_df.pop(checkpoint)
        if self._verbose:
            print_wrapped(
                f'Removed working data checkpoint "{out_chkpt}".',
                type='UPDATE'
            )

        return self


    # --------------------------------------------------------------------------
    # DATAFRAME MANIPULATION + INDEXING
    # --------------------------------------------------------------------------
    def select_vars(self, vars: list[str]):
        """Selects subset of (column) variables in-place on the working 
        train and test DataFrames. 

        Parameters
        ----------
        - vars : list[str]
        """
        self._working_df_test = self._working_df_test[vars]
        self._working_df_train = self._working_df_train[vars]
        self._categorical_vars, self._continuous_vars, \
            self._categorical_to_categories =\
            self._compute_categorical_continuous_vars(self._working_df_train)
        if self._verbose:
            shapes_dict = self._shapes_str_formatted()
            print_wrapped(
                f'Selected columns {list_to_string(vars)}. ' +\
                'Shapes of train, test DataFrames: ' + \
                f'{shapes_dict["train"]}, {shapes_dict["test"]}.',
                type='UPDATE'
            )


    def drop_vars(self, vars: list[str]):
        """Drops subset of variables (columns) in-place on the working 
        train and test DataFrames. 

        Parameters
        ----------
        - vars : list[str]
        """
        self._working_df_test.drop(vars, axis='columns', inplace=True)
        self._working_df_train.drop(vars, axis='columns', inplace=True)
        self._categorical_vars, self._continuous_vars, \
            self._categorical_to_categories =\
            self._compute_categorical_continuous_vars(self._working_df_train)
        if self._verbose:
            shapes_dict = self._shapes_str_formatted()
            print_wrapped(
                f'Dropped columns {list_to_string(vars)}. '+\
                'Shapes of train, test DataFrames: ' + \
                f'{shapes_dict["train"]}, {shapes_dict["test"]}.', 
                type='UPDATE'
            )


    def drop_train_examples(self, indices: list):
        """Drops subset of examples (rows) in-place on the working train 
        DataFrame. 

        Parameters
        ----------
        - indices : list.
        """
        self._working_df_train.drop(indices, axis='index', inplace=True)
        self._categorical_vars, self._continuous_vars, \
            self._categorical_to_categories =\
            self._compute_categorical_continuous_vars(self._working_df_train)
        if self._verbose:
            shapes_dict = self._shapes_str_formatted()
            print_wrapped(
                f'Dropped rows {list_to_string(indices)}. '+\
                'Shapes of train, test DataFrames: ' + \
                f'{shapes_dict["train"]}, {shapes_dict["test"]}.',
                type='UPDATE'
            )


    def set_yvar(self, y_var: str | None):
        """Specifies the y-variable. All other variables are treated as 
        predictors for the y-variable.

        Parameters
        ----------
        - y_var : str. Name of the y-variable.

        Returns
        -------
        - self : DataHandler
        """
        if y_var is None:
            self._yvar = None
            if self._verbose:
                print_wrapped(
                    'Cleared y-variable.',
                    type='UPDATE'
                )
            return self
        if y_var in self._categorical_vars + self._continuous_vars:
            self._yvar = y_var
            if self._verbose:
                print_wrapped(
                    f'Set y-variable to {y_var}.',
                    type='UPDATE'
                )
        else:
            raise ValueError('Invalid y-variable name.')
        return self

        
    def set_Xvars(self, X_vars: list[str] | None):
        """
        Specifies the predictor variables. The y-variable if specified 
        is not included.
        
        Parameters
        ----------
        - X_vars : list[str]. Names of the predictor variables.

        Returns
        -------
        - self : DataHandler
        """
        if X_vars is None:
            X_vars = self.categorical_vars(ignore_yvar=True) +\
                self.continuous_vars(ignore_yvar=True)
        if self._yvar is not None:
            if self._yvar in X_vars:
                X_vars.remove(self._yvar)
        self._Xvars = X_vars
        if self._verbose:
            print_wrapped(
                f'Set predictor variables to {list_to_string(X_vars)}.',
                type='UPDATE'
            )
        return self
    



    def force_binary(self, vars: list[str], pos_label: str = None, 
                     ignore_multiclass: bool = False):
        """Forces variables to be binary (0 and 1 valued continuous variables). 
        Does nothing if the data contains more than two classes unless 
        ignore_multiclass is True and pos_label is specified, 
        in which case all classes except pos_label are labeled with zero.

        Parameters
        ----------
        - vars : list[str]. Name of variables.
        - pos_label : str. Default: None. The positive label. If None, the 
            first class is the positive label.
        - ignore_multiclass : bool. Default: False. If True, all classes 
            except pos_label are labeled with zero.
        
        Returns
        -------
        - self : DataHandler
        """
        if pos_label is None and ignore_multiclass:
            raise ValueError(
                'pos_label must be specified if ignore_multiclass is True.')
        

        success_vars = []
        for var in vars:
            if var not in self._working_df_train.columns:
                raise ValueError(f'Invalid variable name: {var}.')
            
            if pos_label is None:
                unique_vals = self._working_df_train[var].unique()
                if len(unique_vals) > 2:
                    if self._verbose:
                        print_wrapped(
                            'More than two classes present for ' +\
                            f'{var}. Skipping {var}.',
                            type='WARNING'
                        )
                        continue
                pos_label = unique_vals[0]
                self._working_df_train[var] = \
                    self._working_df_train[var].apply(
                        lambda x: 1 if x == pos_label else 0)
                self._working_df_test[var] = \
                    self._working_df_test[var].apply(
                        lambda x: 1 if x == pos_label else 0)
            else:
                unique_vals = self._working_df_train[var].unique()
                if len(unique_vals) > 2:
                    if not ignore_multiclass:
                        if self._verbose:
                            print_wrapped(
                                'More than two classes present for ' +\
                                f'{var}. Skipping {var}.',
                                type='WARNING'
                            )
                        continue
                self._working_df_train[var] = \
                    self._working_df_train[var].apply(
                        lambda x: 1 if x == pos_label else 0)
                self._working_df_test[var] = \
                    self._working_df_test[var].apply(
                        lambda x: 1 if x == pos_label else 0)
                
            success_vars.append(var)
            
        if self._verbose:
            if len(success_vars) == 0:
                print_wrapped(
                    'No variables were forced to binary.',
                    type='WARNING'
                )
            else:
                colored_text = color_text(
                    list_to_string(success_vars), "purple")
                print_wrapped(
                    f'Forced variables {colored_text} to binary.',
                    type='UPDATE'
                )

        self._categorical_vars, self._continuous_vars, \
            self._categorical_to_categories =\
            self._compute_categorical_continuous_vars(self._working_df_train)
        return self



    def force_categorical(self, vars: list[str]):
        """Forces variables to become categorical. 
        Example use case: prepare numerically-coded categorical variables 

        Parameters
        ----------
        - vars : list[str]

        Returns
        -------
        - self : DataHandler
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
                f'Forced variables {list_to_string(vars)} to categorical.',
                type='UPDATE'
            )

        self._categorical_vars, self._continuous_vars, \
            self._categorical_to_categories =\
            self._compute_categorical_continuous_vars(self._working_df_train)
        return self




    # --------------------------------------------------------------------------
    # GETTERS
    # --------------------------------------------------------------------------

    def df_all(self, onehotted: bool = False, dropfirst: bool = True):
        """Returns the working train and test DataFrames concatenated.

        Parameters
        ----------
        - onehotted : bool. Default: False. 
            If True, returns the one-hot encoded DataFrame.
        - dropfirst : bool. Default: True.
            If True, the first dummy variable is dropped. Does nothing if
            onehotted is False.

        Returns
        -------
        - pd.DataFrame
        """
        no_test = True

        for a, b in zip(self._working_df_train.index, 
                        self._working_df_test.index):
            if a != b:
                no_test = False
                break


        if no_test:
            out = self.df_train(onehotted, dropfirst)
        else:
            out = pd.concat([self.df_train(onehotted, dropfirst), 
                            self.df_test(onehotted, dropfirst)])
        return out


    def df_train(self, onehotted: bool = False, dropfirst: bool = True):
        """Returns the working train DataFrame.

        Parameters
        ----------
        - onehotted : bool. Default: False. 
            If True, returns the one-hot encoded DataFrame.
        - dropfirst : bool. Default: True.
            If True, the first dummy variable is dropped. Does nothing if
            onehotted is False.

        Returns
        -------
        - pd.DataFrame
        """
        if onehotted:
            return self._onehot(self._working_df_train, dropfirst=dropfirst)
        return self._working_df_train


    def df_test(self, onehotted: bool = False, dropfirst: bool = True):
        """Returns the working test DataFrame.

        Parameters
        ----------
        - onehotted : bool. Default: False. 
            If True, returns the one-hot encoded DataFrame.
        - dropfirst : bool. Default: True.
            If True, the first dummy variable is dropped. Does nothing if
            onehotted is False.

        Returns
        -------
        - pd.DataFrame
        """
        if onehotted:
            return self._onehot(self._working_df_test, dropfirst=dropfirst)
        return self._working_df_test
    

    def df_train_split(self, 
                       onehotted: bool = True, 
                       dropfirst: bool = True, 
                       dropna: bool = True):
        """Returns the working train DataFrame subsetted by the predictors 
            and the y-variable.

        Parameters
        ----------
        - onehotted : bool. Default: True. 
            If True, returns the one-hot encoded DataFrame.
        - dropfirst : bool. Default: True.
            If True, the first dummy variable is dropped. Does nothing if
            onehotted is False.
        - dropna : bool. Default: True.
            If True, drops rows with missing values.

        Returns
        -------
        - pd.DataFrame
        - pd.Series
        """
        out_train = self._working_df_train[self._Xvars + [self._yvar]]

        if onehotted:
            out_train = self._onehot(out_train, dropfirst=dropfirst, 
                                     fit=True, ignore_yvar=True)

        if dropna:
            prev_len = len(out_train)
            out_train = out_train.dropna()
            if self._verbose and (prev_len - len(out_train)) / prev_len > 0.2:
                print_wrapped(
                    f'Dropped {prev_len - len(out_train)} of {prev_len} ' +\
                    'total rows.',
                    type='WARNING'
                )

        return (out_train.drop(self._yvar, axis=1), out_train[self._yvar])
    

    def df_test_split(self, onehotted: bool = True, 
                       dropfirst: bool = True, dropna: bool = True):
        """Returns the working test DataFrame subsetted by the predictors 
            and the y-variable.

        Parameters
        ----------
        - onehotted : bool. Default: True. 
            If True, returns the one-hot encoded DataFrame.
        - dropfirst : bool. Default: True.
            If True, the first dummy variable is dropped. Does nothing if
            onehotted is False.
        - dropna : bool. Default: True.
            If True, drops rows with missing values.
            
        Returns
        -------
        - pd.DataFrame
        - pd.Series
        """
        out_test = self._working_df_test[self._Xvars + [self._yvar]]

        if onehotted:
            out_test = self._onehot(out_test, dropfirst=dropfirst, 
                                    fit=False, ignore_yvar=True)

        if dropna:
            prev_len = len(out_test)
            out_test = out_test.dropna()
            if self._verbose and (prev_len - len(out_test)) / prev_len > 0.2:
                print_wrapped(
                    f'Dropped {prev_len - len(out_test)} of {prev_len} ' +\
                    'total rows.',
                    type='WARNING'
                )

        return (out_test.drop(self._yvar, axis=1), out_test[self._yvar])
    

    def vars(self, ignore_yvar: bool = True):
        """Returns a list of all variables in the working DataFrames
        
        Parameters
        ----------
        - ignore_yvar : bool. Default: True.
        """
        out = self._working_df_train.columns.to_list()
        if ignore_yvar and self._yvar is not None:
            if self._yvar in out:
                out.remove(self._yvar)
        return out
    

    def Xvars(self) -> str | None:
        """Returns copy of list of predictor variables.
        
        Returns
        -------
        - str | None. None if no predictor variables are set.
        """
        return self._Xvars.copy()
    

    def yvar(self) -> str | None:
        """Returns the y-variable.
        
        Returns
        -------
        - str | None. None if no y-variable is set.
        """
        return self._yvar


    def continuous_vars(self, ignore_yvar: bool = True):
        """Returns copy of list of continuous variables.
        
        Parameters
        ----------
        - ignore_yvar : bool. Default: True.
        """
        out = self._continuous_vars.copy()
        if ignore_yvar and self._yvar is not None:
            if self._yvar in out:
                out.remove(self._yvar)
        return out

    def categorical_vars(self, ignore_yvar: bool = True):
        """Returns copy of list of categorical variables.
        
        Parameters
        ----------
        - ignore_yvar : bool. Default: True.
        """
        out = self._categorical_vars.copy()
        if ignore_yvar and self._yvar is not None:
            if self._yvar in out:
                out.remove(self._yvar)
        return out


    def head(self, n = 5):
        """Returns the first n rows of the working train DataFrame."""
        return self._working_df_test.head(n)
    

    def yscaler(self) -> BaseSingleVarScaler | None:
        """Returns the y-variable scaler, which could be None."""
        if self._yvar is None:
            return None
        return self._continuous_var_to_scaler[self._yvar]
    

    def scaler(self, var: str) -> BaseSingleVarScaler | None:
        """Returns the scaler for a continuous variable, which could be None.
        
        Parameters
        ----------
        - var : str
        """
        return self._continuous_var_to_scaler[var]
        

    def copy(self, 
             y_var: str = None, 
             X_vars: str = None, 
             verbose: bool = False) -> 'DataHandler':
        """Creates a shallow copy of the DataHandler object. That is, 
        the returned object will be initialized with the current 
        working DataFrames, but will not have any other checkpoints saved.
        The scalers dataframe will be preserved. 
        
        Parameters
        ----------
        - y_var : str | None. Default: None.
            If None, the y-variable is not changed from the parent object.
        - X_vars : list[str] | None. Default: None.
            If None, the predictor variables are not changed from the parent
            object.
        - verbose : bool. Default: False.

        Returns
        -------
        - DataHandler
        """
        if y_var is None:
            y_var = self._yvar
        if X_vars is None:
            X_vars = self._Xvars

        new = DataHandler(
            self._working_df_train, 
            self._working_df_test, 
            y_var=y_var, X_vars=X_vars,
            verbose=verbose, name=self._name + '_copy')
        new._continuous_var_to_scaler = self._continuous_var_to_scaler.copy()

        return new


    def kfold_copies(self, 
                     k: int, 
                     y_var: str = None, 
                     X_vars: str = None, 
                     shuffle: bool = True, 
                     seed: int = 42,
                     verbose: bool = False) -> list['DataHandler']:
        """Returns k shallow copies of the DataHandler object, each with the 
        properly specified train and test DataFrames for k-fold cross 
        validation.

        The scalers will be preserved. 
        
        Parameters
        ----------
        - k : int. Number of folds.
        - y_var : str | None. Default: None.
            If None, the y-variable is not changed from the parent object.
        - X_vars : list[str] | None. Default: None.
            If None, the predictor variables are not changed from the parent
            object.
        - shuffle : bool. Default: True. If True, shuffles the examples 
            before splitting into folds.
        - seed : int. Default: 42. Random seed for shuffling.
        - verbose : bool. Default: False.

        Returns 
        -------
        - list[DataHandler]
        """

        if y_var is None:
            y_var = self._yvar
        if X_vars is None:
            X_vars = self._Xvars

        cv = KFold(n_splits=k, shuffle=shuffle, random_state=seed)
        out = []
        for i, (train_index, test_index) in enumerate(
            cv.split(self._working_df_train)):
            train_df = self._working_df_train.iloc[train_index]
            test_df = self._working_df_train.iloc[test_index]
            new = DataHandler(
                train_df, test_df, 
                verbose=verbose, y_var=y_var, 
                X_vars=X_vars, name=self._name + f'_fold_{i}',
            )
            new._continuous_var_to_scaler =\
                self._continuous_var_to_scaler.copy()
            out.append(new)
        return out
    

    # --------------------------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------------------------

    def onehot(self,
               vars: list[str] = None,
               dropfirst: bool = True) -> 'DataHandler':
        """One-hot encodes all categorical variables in-place.
        
        Parameters
        ----------
        - vars : list[str]. Default: None.
            If not None, only one-hot encodes the specified variables.
        - dropfirst : bool. Default: True. 
            If True, the first dummy variable is dropped. 

        Returns
        -------
        - self : DataHandler
        """
        if vars is None:
            vars = self.categorical_vars(ignore_yvar=False)
        self._working_df_train = self._onehot(self._working_df_train, 
                                              vars=vars,
                                              dropfirst=dropfirst, 
                                              ignore_yvar=False,
                                              fit=True)
        self._working_df_test = self._onehot(self._working_df_test,
                                             vars=vars,
                                             dropfirst=dropfirst, 
                                             ignore_yvar=False,
                                             fit=False)

        self._categorical_vars, self._continuous_vars, \
            self._categorical_to_categories =\
            self._compute_categorical_continuous_vars(self._working_df_train)        

        return self



    def drop_highly_missing_vars(self, 
                                 threshold: float = 0.5) -> 'DataHandler':
        """Drops columns with more than 50% missing values (on train) in-place.
        
        Parameters
        ----------
        - threshold : float. Default: 0.5. Proportion of missing values
            above which a column is dropped.

        Returns
        -------
        - self : DataHandler
        """
        prev_vars = self._working_df_train.columns.to_list()
        self._working_df_train.dropna(axis=1, 
            thresh=threshold*len(self._working_df_train),
            inplace=True)
        curr_vars = self._working_df_train.columns.to_list()
        vars_dropped = set(prev_vars) - set(curr_vars)

        self._working_df_test.drop(vars_dropped, 
            axis=1, inplace=True)
        
        if self._verbose:
            print_wrapped(
                f'Dropped variables {list_to_string(vars_dropped)} ' +\
                f'with more than {threshold*100}% ' +\
                'missing values.',
                type='UPDATE'
            )

        self._categorical_vars, self._continuous_vars, \
            self._categorical_to_categories =\
            self._compute_categorical_continuous_vars(self._working_df_train)

        return self


    def dropna_yvar(self) -> 'DataHandler':
        """Drops rows with missing values in the y-variable in-place.
        
        Returns
        -------
        - self : DataHandler
        """
        if self._yvar is not None:
            self.dropna(vars=[self._yvar])
        return self


    def dropna(self, vars: list[str] = None) -> 'DataHandler':
        """Drops rows with missing values in-place.

        Parameters
        ----------
        - vars : list[str]. Default: None.
            Subset of variables (columns). 
            If not None, only considers missing values in the subset of 
            variables.

        Returns
        -------
        - self : DataHandler
        """
        self._working_df_train.dropna(subset=vars, inplace=True)
        self._working_df_test.dropna(subset=vars, inplace=True)
        if self._verbose:
            shapes_dict = self._shapes_str_formatted()
            print_wrapped(
                'Dropped rows with missing values. ' +\
                'Shapes of train, test DataFrames: ' + \
                f'{shapes_dict["train"]}, {shapes_dict["test"]}.',
                type='UPDATE'
            )
        return self



    def impute(self, 
               continuous_strategy: Literal['median', 'mean', '5nn'] = 'median', 
               categorical_strategy: Literal['most_frequent'] = 'most_frequent', 
               ignore_yvar: bool = True) -> 'DataHandler':
        """Imputes missing values in-place.
        
        Parameters
        ----------
        - continuous_strategy : Literal['median', 'mean', '5nn']. 
            Default: 'median'.
            Strategy for imputing missing values in continuous variables.
            - 'median': impute with median.
            - 'mean': impute with mean.
            - '5nn': impute with 5-nearest neighbors.
        - categorical_strategy : Literal['most_frequent']. 
            Default: 'most_frequent'.
            Strategy for imputing missing values in categorical variables.
            - 'most_frequent': impute with most frequent value.
        - ignore_yvar : bool. Default: True. If True, the y-variable is not 
            imputed.

        Returns
        -------
        - self : DataHandler
        """
        continuous_vars = self.continuous_vars(ignore_yvar=ignore_yvar)
        categorical_vars = self.categorical_vars(ignore_yvar=ignore_yvar)

        # impute continuous variables
        if len(continuous_vars) > 0:
            if continuous_strategy == '5nn':
                imputer = KNNImputer(n_neighbors=5, keep_empty_features=True)
            else:
                imputer = SimpleImputer(strategy=continuous_strategy, 
                                        keep_empty_features=True)
            self._working_df_train[continuous_vars] =\
                imputer.fit_transform(self._working_df_train[continuous_vars])
            self._working_df_test[continuous_vars] =\
                imputer.transform(self._working_df_test[continuous_vars])
        
        # impute categorical variables
        if len(categorical_vars) > 0:
            imputer = SimpleImputer(strategy=categorical_strategy, 
                                    keep_empty_features=True)
            self._working_df_train[categorical_vars] =\
                imputer.fit_transform(
                    self._working_df_train[categorical_vars])
            self._working_df_test[categorical_vars] =\
                imputer.transform(self._working_df_test[categorical_vars])
            
            if self._verbose:
                print_wrapped(
                    'Imputed missing values with ' +\
                    f'continuous strategy "{continuous_strategy}", ' +\
                    f'categorical strategy "{categorical_strategy}".',
                    type='UPDATE'
                )

        return self



    def scale(self, 
              vars: list[str] = None, 
              strategy: Literal['standardize', 'minmax', 'log', 
                                'log1p'] = 'standardize') -> 'DataHandler':
        """Scales variables in-place.

        Parameters
        ----------
        - vars : list[str]. Default: None.
            Subset of variables (columns). 
            If None, scales all continuous variables.
        - strategy : Literal['standardize', 'minmax', 'log', 'log1p']. 
            Default: 'standardize'.
            Strategy for scaling continuous variables.
            - 'standardize': standardize variables.
            - 'minmax': min-max scale variables.
            - 'log': log transform variables.
            - 'log1p': log1p transform variables.

        Returns
        -------
        - self : DataHandler
        """
        if vars is None:
            vars = self._continuous_vars
        
        for var in vars:
            train_data = self._working_df_train[var].to_numpy()
            if strategy == 'standardize':
                scaler = StandardizeSingleVar(var, train_data)
            elif strategy == 'minmax':
                scaler = MinMaxSingleVar(var, train_data)
            elif strategy == 'log':
                scaler = LogTransformSingleVar(var, train_data)
            elif strategy == 'log1p':
                scaler = Log1PTransformSingleVar(var, train_data)
            else:
                raise ValueError('Invalid scaling strategy.')
            
            self._working_df_train[var] = scaler.transform(
                self._working_df_train[var].to_numpy())
            self._working_df_test[var] = scaler.transform(
                self._working_df_test[var].to_numpy())
            self._continuous_var_to_scaler[var] = scaler

        if self._verbose:
            print_wrapped(
                f'Scaled variables {list_to_string(vars)} ' + \
                    f'using strategy "{strategy}".',
                type='UPDATE'
            )
        return self

    # --------------------------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------------------------
    def _verify_input_dfs(self, df1: pd.DataFrame, df2: pd.DataFrame):
        """Ensures that the original train and test DataFrames have the 
        same variables. 

        Parameters
        ----------
        - df1 : pd.DataFrame
        - df2 : pd.DataFrame
        """
        l1 = set(df1.columns.to_list())
        l2 = set(df2.columns.to_list())
        if len(l1.union(l2)) != len(l1):
            raise RuntimeWarning('The train and test DataFrames' + \
                ' do not have the same variables.')


    def _compute_categorical_continuous_vars(self, df: pd.DataFrame):
        """Returns the categorical and continuous column values. 
        Also returns the categorical variables mapped to their categories.
        
        Parameters
        ----------
        - df : pd.DataFrame

        Returns
        -------
        - categorical_vars : list[str]
        - continuous_vars : list[str]
        - c
        """
        categorical_vars = df.select_dtypes(
            include=['object', 'category', 'bool']).columns.to_list()
        continuous_vars = df.select_dtypes(
            exclude=['object', 'category', 'bool']).columns.to_list()
        categoical_mapped = self._compute_categories(df, categorical_vars)
        return categorical_vars, continuous_vars, categoical_mapped
        

    def _force_train_test_var_agreement(
            self, 
            df_train: pd.DataFrame, 
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
                    f'Columns {list_to_string(extra_test_columns)} not ' +\
                    'in train have been dropped from test.',
                    type='WARNING'
                )
            df_test.drop(columns=extra_test_columns, axis=1, inplace=True)
        if len(missing_test_columns) > 0:
            if self._verbose:
                print_wrapped(
                    f'Columns {list_to_string(missing_test_columns)} not ' +\
                    'in test have been added to test with 0-valued entries.',
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
    


    def _shapes_str_formatted(self):
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
            'train': color_text(str(self._working_df_train.shape), 
                                color='yellow'),
            'test': color_text(str(self._working_df_test.shape),
                               color='yellow')
        }
    




    def _compute_categories(self, 
                            df: pd.DataFrame,
                            categorical_vars: list[str]):
        """Returns a dictionary containing the categorical variables 
        each mapped to a list of all categories in the variable.
        
        Parameters
        ----------
        - df : pd.DataFrame.
        - categorical_vars : list[str]. 

        Returns
        -------
        - dict
        """
        categories_dict = {}
        for var in categorical_vars:
            categories_dict[var] = df[var].unique().tolist()
        return categories_dict




    def _onehot(self, 
                df: pd.DataFrame, 
                vars: list[str] = None,
                dropfirst: bool = True, 
                ignore_yvar: bool = True, 
                fit: bool = True) -> pd.DataFrame:
        """One-hot encodes all categorical variables with more than 
        two categories. Optionally does not ignore the set y-variable, 
        if specified.
        
        Parameters
        ----------
        - df : pd.DataFrame
        - vars : list[str]. Default: None.
            If not None, only one-hot encodes the specified variables.
        - dropfirst : bool. Default: True. 
            If True, the first dummy variable is dropped.
        - ignore_yvar : bool. Default: True.
            If True, the y-variable (if specified) is not one-hot encoded.
        - fit : bool. Default: True.
            If True, fits the encoder on the training data. Otherwise,
            only transforms the test data.

        Returns
        -------
        - df_train encoded : pd.DataFrame
        """
        if vars is None:
            categorical_vars, _, _ =\
                self._compute_categorical_continuous_vars(df)
        else:
            for var in vars:
                if var not in df.columns:
                    raise ValueError(f'Invalid variable name: {var}')
            categorical_vars = vars


        if ignore_yvar and self._yvar is not None:
            if self._yvar in categorical_vars:
                categorical_vars.remove(self._yvar)


        if categorical_vars:
            if dropfirst:
                drop = 'first'
            else:
                drop = None

            if fit:
                self._onehot_encoder = \
                    OneHotEncoder(drop=drop, sparse_output=False, 
                                  handle_unknown='ignore')
                encoded = self._onehot_encoder.fit_transform(
                    df[categorical_vars])
                feature_names = self._onehot_encoder.get_feature_names_out(
                    categorical_vars)
                df_encoded = pd.DataFrame(
                    encoded, columns=feature_names, index=df.index)
            
            else:
                encoded = ignore_warnings(self._onehot_encoder.transform)(
                    df[categorical_vars])
                feature_names = self._onehot_encoder.get_feature_names_out(
                    categorical_vars)
                df_encoded = pd.DataFrame(
                    encoded, columns=feature_names, index=df.index)
            
            return pd.concat(
                    [
                        df_encoded, 
                        df.drop(columns=categorical_vars)
                    ], 
                    axis=1
                )
        else:
            return df





    def __len__(self):
        """Returns the number of examples in working_df_train."""
        return len(self._working_df_train)



    def __str__(self):
        """Returns a string representation of the DataHandler object."""
        working_df_test = self._working_df_test
        working_df_train = self._working_df_train

        max_width = TOSTR_MAX_WIDTH

        textlen_shapes = len(str(working_df_train.shape) +\
            str(working_df_test.shape)) + 25
        shapes_message_buffer_left = (max_width - textlen_shapes) // 2
        shapes_message_buffer_right = int(np.ceil(
            (max_width - textlen_shapes) / 2))


        shapes_message =\
            color_text(bold_text('Train shape: '), 'none') +\
            color_text(str(working_df_train.shape), 'yellow') + \
            ' '*shapes_message_buffer_left +\
            color_text(bold_text('Test shape: '), 'none') + \
            color_text(str(working_df_test.shape), 'yellow')  + \
            ' '*shapes_message_buffer_right


        title_message = color_text(bold_text(self._name), 'none')
        title_message = fill_ignore_format(title_message, width=max_width)
        
        categorical_message = color_text(
            bold_text('Categorical variables:'), 'none') + '\n'
        categorical_var_message = ''
        if len(self.categorical_vars()) == 0:
            categorical_var_message += color_text('None', 'yellow')
        else:
            categorical_var_message += list_to_string(self.categorical_vars())
        categorical_var_message = fill_ignore_format(categorical_var_message, 
            width=max_width, initial_indent=2, subsequent_indent=2)

        continuous_message = color_text(
            bold_text('Continuous variables:'), 'none') + '\n'
        continuous_var_message = ''
        if len(self.continuous_vars()) == 0:
            continuous_var_message += color_text('None', 'yellow')
        else:
            continuous_var_message += list_to_string(self.continuous_vars())
        continuous_var_message = fill_ignore_format(
            continuous_var_message, width=max_width, 
            initial_indent=2, subsequent_indent=2)

        bottom_divider = '\n' + color_text('='*max_width, 'none')
        divider = '\n' + color_text('-'*max_width, 'none') + '\n'
        divider_invisible = '\n' + ' '*max_width + '\n'
        top_divider = color_text('='*max_width, 'none') + '\n'

        final_message = top_divider + title_message + divider +\
            shapes_message + divider + categorical_message +\
            categorical_var_message + divider_invisible + continuous_message +\
            continuous_var_message + bottom_divider
        
        return final_message


    def _repr_pretty_(self, p, cycle):
        p.text(str(self))




