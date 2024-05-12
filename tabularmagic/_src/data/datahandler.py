import pandas as pd
from typing import Literal
from ..util.console import print_wrapped
from .preprocessing import (BaseSingleVarScaler, Log1PTransformSingleVar, 
    LogTransformSingleVar, MinMaxSingleVar, StandardizeSingleVar)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold



class DataHandler:
    """DataHandler: handles all aspects of data preprocessing and loading.
    """

    def __init__(self, df_train: pd.DataFrame, 
                 df_test: pd.DataFrame, 
                 y_var: str | None = None,
                 X_vars: list[str] | None = None,
                 id: str = None,
                 verbose: bool = True):
        """Initializes a DataHandler object.

        Parameters
        ----------
        - df_train : pd.DataFrame.
        - df_test : pd.DataFrame.
        - y_var : str | None. Default: None.
        - X_vars : list[str] | None. Default: None.
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

        # keep track of scalers
        self._continuous_var_to_scalar = {
            var: None for var in self._continuous_vars
        }

        # set the y-variable
        self.set_yvar(y_var=y_var)
        self.set_Xvars(X_vars=X_vars)


        # set the id
        if id is None:
            self.nickname = 'DataHandler'
        else:
            self.nickname = id


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
            self._working_df_test = self._orig_df_test.copy()
            self._working_df_train = self._orig_df_train.copy()
            if self._verbose:
                shapes_dict = self.shapes()
                print_wrapped(
                    'Working datasets reset to original datasets. ' +\
                    'Shapes of train, test datasets: ' + \
                    f'{shapes_dict["train"]}, {shapes_dict["test"]}.',
                    type='UPDATE'
                )
        else:
            self._working_df_test =\
                self._checkpoint_name_to_df[checkpoint][0].copy()
            self._working_df_train =\
                self._checkpoint_name_to_df[checkpoint][1].copy()
            if self._verbose:
                shapes_dict = self.shapes()
                print_wrapped(
                    'Working datasets reset to checkpoint ' +\
                    f'"{checkpoint}". ' +\
                    'Shapes of train, test datasets: ' + \
                    f'{shapes_dict["train"]}, {shapes_dict["test"]}.', 
                    type='UPDATE'
                )
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
            self._working_df_test.copy(),
            self._working_df_train.copy()
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
        self._working_df_test = self._working_df_test[vars]
        self._working_df_train = self._working_df_train[vars]
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
        self._working_df_test.drop(vars, axis='columns', inplace=True)
        self._working_df_train.drop(vars, axis='columns', inplace=True)
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
                    f'Set y-variable to "{y_var}".',
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
                f'Set predictor variables to "{X_vars}".',
                type='UPDATE'
            )
        return self


    # --------------------------------------------------------------------------
    # GETTERS
    # --------------------------------------------------------------------------
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
    

    def df_train_split(self, onehotted: bool = True, 
                       dropfirst: bool = True, dropna: bool = True):
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
            out_train.dropna(inplace=True)
            if self._verbose:
                print_wrapped(
                    f'Dropped {prev_len - len(out_train)} rows with missing ' +\
                        'values.',
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
            out_test.dropna(inplace=True)
            if self._verbose:
                print_wrapped(
                    f'Dropped {prev_len - len(out_test)} rows with missing ' +\
                        'values.',
                    type='WARNING'
                )

        return (out_test.drop(self._yvar, axis=1), out_test[self._yvar])
    

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
            'train': self._working_df_test.shape,
            'test': self._working_df_test.shape
        }
    

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
        return self._continuous_var_to_scalar[self._yvar]
    

    def scaler(self, var: str) -> BaseSingleVarScaler | None:
        """Returns the scaler for a continuous variable, which could be None.
        
        Parameters
        ----------
        - var : str
        """
        return self._continuous_var_to_scalar[var]
        

    def copy(self, y_var: str = None, 
             X_vars: str = None, verbose: bool = False) -> 'DataHandler':
        """Creates a shallow copy of the DataHandler object. That is, 
        the returned object will be initialized with the current 
        working DataFrames, but will not have any other checkpoints saved.
        No other attributes such as y-variable and predictor specifications
        are retained.
        
        Parameters
        ----------
        - y_var : str | None. Default: None.
        - X_vars : list[str] | None. Default: None.
        - verbose : bool. Default: False.

        Returns
        -------
        - DataHandler
        """
        return DataHandler(
            self._working_df_train, 
            self._working_df_test, 
            y_var=y_var, X_vars=X_vars,
            verbose=verbose, id=self.nickname + '_copy')
    

    def kfold_copies(self, k: int, y_var: str = None, X_vars: str = None, 
                     shuffle: bool = True, 
                     seed: int = 42,
                     verbose: bool = False) -> list['DataHandler']:
        """Returns k shallow copies of the DataHandler object, each with the 
        properly specified train and test datasets for k-fold cross 
        validation.
        
        Parameters
        ----------
        - k : int. Number of folds.
        - y_var : str | None. Default: None.
        - X_vars : list[str] | None. Default: None.
        - shuffle : bool. Default: True. If True, shuffles the examples 
            before splitting into folds.
        - seed : int. Default: 42. Random seed for shuffling.
        - verbose : bool. Default: False.

        Returns 
        -------
        - list[DataHandler]
        """
        cv = KFold(n_splits=k, shuffle=shuffle, random_state=seed)
        out = []
        for i, (train_index, test_index) in enumerate(
            cv.split(self._working_df_train)):
            train_df = self._working_df_train.iloc[train_index]
            test_df = self._working_df_train.iloc[test_index]
            out.append(DataHandler(
                train_df, test_df, 
                verbose=verbose, y_var=y_var, 
                X_vars=X_vars, id=self.nickname + f'_fold_{i}'))
        return out
    

    # --------------------------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------------------------

    def force_categorical(self, vars: list[str]):
        """Forces variables to become categorical. 
        Example use case: prepare numerically-coded categorical variables 

        Parameters
        ----------
        - vars : list[str]
        """
        self._working_df_train[vars] =\
            self._working_df_train[vars].astype('str')
        self._working_df_test[vars] =\
            self._working_df_test[vars].astype('str')
        if self._verbose:
            print_wrapped(
                f'Converted variables {vars} to categorical.',
                type='UPDATE'
            )
        self._categorical_vars, self._continuous_vars =\
            self._compute_categorical_continuous_vars(self._working_df_train)
        


    def _onehot(self, df: pd.DataFrame, 
                dropfirst: bool = True, 
                ignore_yvar: bool = True, 
                fit: bool = True) -> pd.DataFrame:
        """One-hot encodes all categorical variables with more than 
        two categories. Optionally does not ignore the set y-variable, 
        if specified.
        
        Parameters
        ----------
        - df : pd.DataFrame
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
        categorical_vars, _ = self._compute_categorical_continuous_vars(df)

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
                    OneHotEncoder(drop=drop, sparse_output=False)
                encoded = self._onehot_encoder.fit_transform(
                    df[categorical_vars])
                feature_names = self._onehot_encoder.get_feature_names_out(
                    categorical_vars)
                df_encoded = pd.DataFrame(
                    encoded, columns=feature_names, index=df.index)
            
            else:
                encoded = self._onehot_encoder.transform(
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



    def dropna_yvar(self):
        """Drops rows with missing values in the y-variable in-place.
        
        Returns
        -------
        - self : DataHandler
        """
        if self._yvar is not None:
            self.dropna(vars=[self._yvar])
        return self


    def dropna(self, vars: list[str] = None):
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
            shapes_dict = self.shapes()
            print_wrapped(
                'Dropped rows with missing values. ' +\
                'Shapes of train, test datasets: ' + \
                f'{shapes_dict["train"]}, {shapes_dict["test"]}.',
                type='UPDATE'
            )
        return self



    def impute(self, 
            continuous_strategy: Literal['median', 'mean', '5nn'] = 'median', 
            categorical_strategy: Literal['most_frequent'] = 'most_frequent', 
            ignore_yvar: bool = True):
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
        imputer = SimpleImputer(strategy=categorical_strategy, 
                                keep_empty_features=True)
        self._working_df_train[categorical_vars] =\
            imputer.fit_transform(
                self._working_df_train[categorical_vars])
        self._working_df_test[categorical_vars] =\
            imputer.transform(self._working_df_test[categorical_vars])
        
        if self._verbose:
            print_wrapped(
                'Imputed missing values with strategies: ' +\
                f'continuous: "{continuous_strategy}", ' +\
                f'categorical: "{categorical_strategy}".',
                type='UPDATE'
            )
        return self


    def scale(self, vars: list[str] = None, 
              strategy: Literal['standardize', 'minmax', 'log', 
                                'log1p'] = 'standardize'):
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
        train_data = self._working_df_test[vars].to_numpy()
        for var in vars:
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
                self._working_df_train[var])
            self._working_df_test[var] = scaler.transform(
                self._working_df_test[var])
            if var == self._yvar:
                self._yvar_scaler = scaler
        if self._verbose:
            print_wrapped(
                f'Scaled variables {vars} using strategy "{strategy}".',
                type='UPDATE'
            )
        return self

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
            df_test.drop(columns=extra_test_columns, axis=1, inplace=True)
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



    def __str__(self):
        """Returns a string representation of the DataHandler object."""
        return f'DataHandler object: {self.nickname}'


    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

