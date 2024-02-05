import pandas as pd
import numpy as np




class MinMaxSingleVar():
    """Min max scaling of a single variable"""

    def __init__(self, var_name: str, x: np.ndarray):
        self.var_name = var_name
        self.x = x

    def fit(self):
        self.min = self.x.min()
        self.max = self.x.max()

    def transform(self, x: np.ndarray):
        return (x - self.min) / (self.max - self.min)
        
    def fit_transform(self, x: np.ndarray = None):
        if x is not None:
            self.x = x
        self.fit(self)
        self.transform(self.x)

    def inverse_transform(self, x_scaled: np.ndarray):
        return (self.max - self.min) * x_scaled + self.min




class StandardizeSingleVar():
    """Standard scaling of a single variable"""

    def __init__(self, var_name: str, x: np.ndarray):
        self.var_name = var_name
        self.x = x

    def fit(self):
        self.sigma = self.x.std()
        self.mu = self.x.mean()

    def transform(self, x: np.ndarray):
        return (x - self.mu) / self.sigma
        
    def fit_transform(self, x: np.ndarray = None):
        if x is not None:
            self.x = x
        self.fit(self)
        self.transform(self.x)

    def inverse_transform(self, x_scaled: np.ndarray):
        return self.sigma * x_scaled + self.mu



class DataPreprocessor():
    """Automatic handling of one-hot-encoding, scaling, feature selection, 
    and missing data imputation. 
    """

    def __init__(self, df: pd.DataFrame,
                 onehot_vars: list[str] = [],
                 standardize_vars: list[str] = [],
                 minmax_vars: list[str] = [], 
                 dropfirst_onehot: bool = False):
        """Initializes a DataPreprocessor object. 

        Parameters
        ----------
        - df : pd.DataFrame
        - onehot_vars : list[str]. 
        - standard_scale_vars : list[str].
        - minmax_scale_vars : list[str].
        - dropfirst_onehot : bool. 
        
        Returns
        -------
        - pd.DataFrame
        """
        self.orig_df = df.copy()
        self._onehot_vars = onehot_vars.copy()
        self._dropfirst_onehot = dropfirst_onehot
        self._onehot_encoded_vars = []
        self._standard_scale_vars = standardize_vars.copy()
        self._minmax_scale_vars = minmax_vars.copy()
        return self.forward()

    def forward(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Outputs a preprocessed version of the input DataFrame.

        Parameters
        ----------
        - df : pd.DataFrame
            Default: None. If None, acts upon the DataFrame inputted at 
            initialization.

        Returns
        -------
        - pd.DataFrame
        """
        if df is None:
            df = self.orig_df.copy()
        self._verify_variable_name_validity(df, [self._onehot_vars, 
            self._standard_scale_vars, self._minmax_scale_vars])
        df = self._onehot(df)
        df[self._minmax_scale_vars] = self._minmax_scaler.fit_transform(
            df[self._minmax_scale_vars])
        df[self._standard_scale_vars] = self._standard_scaler.fit_transform(
            df[self._standard_scale_vars])
        return df

    def backward(self, df: pd.DataFrame) -> pd.DataFrame: 
        """Given a DataFrame that is in the preprocessed space, outputs 
        a DataFrame in the original space. 
        
        Parameters
        ----------
        - df : pd.DataFrame

        Returns
        -------
        - pd.DataFrame
        """
        df = df.copy()
        self._verify_variable_name_validity(df, [self._onehot_encoded_vars, 
            self._standard_scale_vars, self._minmax_scale_vars])
        df = self._inverse_onehot(df)
        df[self._minmax_scale_vars] = self._minmax_scaler.inverse_transform(
            df[self._minmax_scale_vars])
        df[self._standard_scale_vars] = self._standard_scaler.inverse_transform(
            df[self._standard_scale_vars])
        return df
    
    def _onehot(self, df: pd.DataFrame):
        """One hot encodes the DataFrame. 

        Parameters
        ----------
        - df : pd.DataFrame

        Returns
        -------
        - pd.DataFrame
        """
        for var in self._onehot_vars:
            temp = pd.get_dummies(df[var], drop_first=self._dropfirst_onehot, 
                                copy=True)
            self._onehot_encoded_vars.append(df.columns.to_list())
            df = pd.concat([df.drop(var, axis=1), temp], axis=1)
        return df

    def _inverse_onehot(self, df: pd.DataFrame):
        """Undo for one hot encoding. 

        Parameters
        ----------
        - df : pd.DataFrame

        Returns
        -------
        - pd.DataFrame
        """
        for encoded_vars, var in zip(self._onehot_encoded_vars, 
            self._onehot_vars):
            orig_column = pd.Series(df[self._onehot_encoded_vars].\
                idxmax(axis=1), name=var)
            df = pd.concat([df.drop(encoded_vars, axis=1), orig_column], 
                            axis=1)
            return df
        
    def _minmax(self, df: pd.DataFrame):
        """Min max scales encodes the DataFrame. 

        Parameters
        ----------
        - df : pd.DataFrame

        Returns
        -------
        - pd.DataFrame
        """
        for var in self._minmax_scale_vars:
            pass


    def _verify_variable_name_validity(self, df: pd.DataFrame, 
                                       list_of_var_lists: list[list[str]]):
        """Verifies that all variable names exist in the DataFrame. 

        Parameters
        ----------
        - df : pd.DataFrame

        Returns
        -------
        - None
        """
        variables = df.columns.to_list()
        for var_list in list_of_var_lists:
            for var in var_list:
                if var not in variables:
                    raise ValueError(f'Invalid input: df.',
                        f'Variable "{var}" is not in df.')








