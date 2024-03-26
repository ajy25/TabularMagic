import pandas as pd
import numpy as np
from typing import Literal, Callable
from sklearn.impute import KNNImputer, SimpleImputer


class BaseSingleVarScaler():

    def __init__(self, var_name: str, x: np.ndarray):
        self.var_name = var_name
        self.x = x[~np.isnan(x)]
        self.fit()

    def fit(self):
        pass

    def transform(self, x: np.ndarray):
        """Transforms x. Robust to missing values in x."""
        pass

    def inverse_transform(self, x_scaled: np.ndarray):
        """Inverse transforms x_scaled. Robust to missing values in x_scaled.
        """
        pass


class MinMaxSingleVar(BaseSingleVarScaler):
    """Min max scaling of a single variable"""

    def __init__(self, var_name: str, x: np.ndarray):
        super().__init__(var_name, x)

    def fit(self):
        self.min = self.x.min()
        self.max = self.x.max()

    def transform(self, x: np.ndarray):
        """Transforms x. Robust to missing values in x."""
        return (x - self.min) / (self.max - self.min)

    def inverse_transform(self, x_scaled: np.ndarray):
        """Inverse transforms x_scaled. Robust to missing values in x_scaled.
        """
        return (self.max - self.min) * x_scaled + self.min
    

class StandardizeSingleVar(BaseSingleVarScaler):
    """Standard scaling of a single variable"""

    def __init__(self, var_name: str, x: np.ndarray):
        super().__init__(var_name, x)

    def fit(self):
        self.sigma = self.x.std()
        self.mu = self.x.mean()

    def transform(self, x: np.ndarray):
        """Transforms x. Robust to missing values in x."""
        return (x - self.mu) / self.sigma

    def inverse_transform(self, x_scaled: np.ndarray):
        """Inverse transforms x_scaled. Robust to missing values in x_scaled.
        """
        return self.sigma * x_scaled + self.mu
    

class LogTransformSingleVar(BaseSingleVarScaler):
    """Log (base e) transform scaling of a single variable"""

    def __init__(self, var_name: str, x: np.ndarray):
        super().__init__(var_name, x)

    def fit(self):
        pass

    def transform(self, x: np.ndarray):
        """Transforms x. Robust to missing values in x."""
        return np.log(x)

    def inverse_transform(self, x_scaled: np.ndarray):
        """Inverse transforms x_scaled. Robust to missing values in x_scaled.
        """
        return np.exp(x_scaled)
    

class ExpTransformSingleVar(BaseSingleVarScaler):
    """Exp (base e) transform scaling of a single variable"""

    def __init__(self, var_name: str, x: np.ndarray):
        super().__init__(var_name, x)

    def fit(self):
        pass

    def transform(self, x: np.ndarray):
        """Transforms x. Robust to missing values in x."""
        return np.exp(x)

    def inverse_transform(self, x_scaled: np.ndarray):
        """Inverse transforms x_scaled. Robust to missing values in x_scaled.
        """
        return np.log(x_scaled)


class Log1PTransformSingleVar(BaseSingleVarScaler):
    """Log1p transform scaling of a single variable"""

    def __init__(self, var_name: str, x: np.ndarray):
        super().__init__(var_name, x)

    def fit(self):
        pass

    def transform(self, x: np.ndarray):
        """Transforms x. Robust to missing values in x."""
        return np.log1p(x)

    def inverse_transform(self, x_scaled: np.ndarray):
        """Inverse transforms x_scaled. Robust to missing values in x_scaled.
        """
        return np.expm1(x_scaled)

    

class CustomFunctionSingleVar(BaseSingleVarScaler):
    """Custom scaling of a single variable"""

    def __init__(self, var_name: str, x: np.ndarray, f: Callable, 
                 f_inv: Callable):
        """
        Parameters
        ----------
        - var_name : str.
        - x : np.ndarray ~ (n_examples,).
        - f : function. 
        f must have one argument, an 1d np.ndarray, and return an np.ndarray 
        of the same size. 
        - f_inv : function. 
        Inverse function of f. 
        """
        self.var_name = var_name
        self.x = x[~np.isnan(x)]
        self.f = f
        self.f_inv = f_inv
        self.fit()

    def fit(self):
        pass

    def transform(self, x: np.ndarray):
        return self.f(x)
    
    def inverse_transform(self, x_scaled: np.ndarray):
        return self.f_inv(x_scaled)





class DataPreprocessor():
    """Automatic handling of one-hot-encoding, scaling, feature selection, 
    and missing data imputation. 

    The original order of the columns are not guaranteed to be maintained. 
    """


    def __init__(self, df: pd.DataFrame,
                 onehot_vars: list[str] = [],
                 standardize_vars: list[str] = [],
                 minmax_vars: list[str] = [], 
                 log1p_vars: list[str] = [],
                 log_vars: list[str] = [],
                 imputation_strategy: Literal[None, 'drop', 'mean', 
                        'median', '5nn', '10nn'] = None,
                 dropfirst_onehot: bool = False):
        """Initializes a DataPreprocessor object. 

        Parameters
        ----------
        - df : pd.DataFrame.
        - onehot_vars : list[str]. 
        - standard_scale_vars : list[str].
        - minmax_scale_vars : list[str].
        - log1p_vars : list[str].
        - log_vars : list[str]
        - imputation_strategy : Literal[None, 'drop', 'mean', 
            'median', '5nn', '10nn'].
        - dropfirst_onehot : bool. 
        
        Returns
        -------
        - pd.DataFrame
        """
        self.orig_df = df.copy()

        # one hot encoding metadata. 
        self._onehot_vars = onehot_vars.copy()
        self._dropfirst_onehot = dropfirst_onehot
        self._onehot_var_to_encoded = {var: [] for var in self._onehot_vars}
        self._binary_vars = []
        self._transformed_vars = []

        # standard scale metadata
        self._standard_scale_vars = standardize_vars.copy()
        self._standard_scale_var_to_scaler = {var: StandardizeSingleVar(
            var_name=var, x=self.orig_df[var].to_numpy()) for \
            var in self._standard_scale_vars}
        
        # min max scale metadata
        self._minmax_scale_vars = minmax_vars.copy()
        self._minmax_scale_var_to_scaler = {var: MinMaxSingleVar(
            var_name=var, x=self.orig_df[var].to_numpy()) for \
            var in self._minmax_scale_vars}
        
        # log1p scale metadata
        self._log1p_scale_vars = log1p_vars.copy()
        self._log1p_scale_var_to_scaler = {var: Log1PTransformSingleVar(
            var_name=var, x=self.orig_df[var].to_numpy()) \
                for var in self._log1p_scale_vars}
        
        # log scale metadata
        self._log_scale_vars = log_vars.copy()
        self._log_scale_var_to_scaler = {var: LogTransformSingleVar(
            var_name=var, x=self.orig_df[var].to_numpy()) \
                for var in self._log_scale_vars}
        
        # missing data metadata
        self._imputation_strategy = imputation_strategy
        self._imputer = None
        

    def forward(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Forward transformation on an entire DataFrame. 
        
        Parameters
        ----------
        - df : pd.DataFrame.
            Default: None. If None, transforms the original DataFrame given 
            at initialization. 

        Returns
        -------
        - pd.DataFrame. 
        """
        df_temp = df.copy()
        df_temp = self._standardize_forward(df_temp)
        df_temp = self._minmax_forward(df_temp)
        df_temp = self._onehot_forward(df_temp)
        df_temp = self._log1p_forward(df_temp)
        df_temp = self._log_forward(df_temp)
        df_temp = self._handle_missing_values(df_temp)
        return df_temp
        

    def get_single_var_scaler(self, var_name: str) -> BaseSingleVarScaler:
        """Returns the BaseSingleVarScaler object corresponding to 
        var_name.
        
        Parameters
        ----------
        - var_name : str.

       Returns
        -------
        - BaseSingleVarScaler.
        """
        if var_name in self._minmax_scale_vars:
            return self._minmax_scale_var_to_scaler[var_name]
        elif var_name in self._standard_scale_vars:
            return self._standard_scale_var_to_scaler[var_name]
        elif var_name in self._log1p_scale_vars:
            return self._log1p_scale_var_to_scaler[var_name]
        elif var_name in self._log_scale_vars:
            return self._log_scale_var_to_scaler[var_name]
        else:
            return None


    def _onehot_forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """One hot transforms subset of DataFrame. If the variable is binary, 
        then nothing will be done. 
        
        Parameters
        ----------
        - df : pd.DataFrame.

        Returns
        -------
        - pd.DataFrame.
        """
        for var in self._onehot_vars:
            dropfirst = self._dropfirst_onehot
            if len(df[~df[var].isna()][var].unique()) == 2:
                # always drop first if binary
                dropfirst = True
            temp_encoded = pd.get_dummies(df[[var]], 
                drop_first=dropfirst)
            self._onehot_var_to_encoded[var] = temp_encoded.columns.to_list()
            df.drop(columns=var, inplace=True)
            df = pd.concat((df, temp_encoded.astype(int)), axis=1)
        self._transformed_vars = df.columns.to_list()
        return df




    def _standardize_forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes subset of the DataFrame. 
        
        Parameters 
        ----------
        - df : pd.DataFrame.

        Returns
        -------
        - pd.DataFrame.
        """
        for var in self._standard_scale_vars:
            df[var] = self._standard_scale_var_to_scaler[var].transform(
                df[var].to_numpy())
        return df
    

    def _minmax_forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """Min max scales subset of the DataFrame. 
        
        Parameters 
        ----------
        - df : pd.DataFrame.

        Returns
        -------
        - pd.DataFrame.
        """
        for var in self._minmax_scale_vars:
            df[var] = self._minmax_scale_var_to_scaler[var].transform(
                df[var].to_numpy())
        return df
    

    def _log1p_forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """log1p scales subset of the DataFrame.
        
        Parameters
        ----------
        - df : pd.DataFrame.

        Returns
        -------
        - pd.DataFrame.
        """
        for var in self._log1p_scale_vars:
            df[var] = self._log1p_scale_var_to_scaler[var].transform(
                df[var].to_numpy())
        return df
    

    def _log_forward(self, df: pd.DataFrame) -> pd.DataFrame:
        """log scales subset of the DataFrame.
        
        Parameters
        ----------
        - df : pd.DataFrame.

        Returns
        -------
        - pd.DataFrame.
        """
        for var in self._log_scale_vars:
            df[var] = self._log_scale_var_to_scaler[var].transform(
                df[var].to_numpy())
        return df

    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles missing values.
        
        Parameters
        ----------
        - df : pd.DataFrame

        Returns
        -------
        - pd.DataFrame.
        """
        if self._imputation_strategy == '5nn':
            self._imputer = KNNImputer(n_neighbors=5, keep_empty_features=True)
            orig_n_vars = len(df.columns)
            df = pd.DataFrame(self._imputer.fit_transform(df), 
                              columns=df.columns)
            if len(df.columns) != orig_n_vars:
                raise RuntimeError('5-nearest neighbor imputation failed.' + \
                    'Please select another imputation strategy.')
        elif self._imputation_strategy == '10nn':
            self._imputer = KNNImputer(n_neighbors=10, keep_empty_features=True)
            orig_n_vars = len(df.columns)
            df = pd.DataFrame(self._imputer.fit_transform(df), 
                              columns=df.columns)
            if len(df.columns) != orig_n_vars:
                raise RuntimeError('10-nearest neighbor imputation failed.' + \
                    'Please select another imputation strategy.')
        elif self._imputation_strategy == 'median':
            self._imputer = SimpleImputer(strategy='median', 
                                          keep_empty_features=True)
            orig_n_vars = len(df.columns)
            df = pd.DataFrame(self._imputer.fit_transform(df), 
                              columns=df.columns)
            if len(df.columns) != orig_n_vars:
                raise RuntimeError('Median imputation failed.' + \
                    'Please select another imputation strategy.')
        elif self._imputation_strategy == 'mean':
            self._imputer = SimpleImputer(strategy='mean', 
                                          keep_empty_features=True)
            orig_n_vars = len(df.columns)
            df = pd.DataFrame(self._imputer.fit_transform(df), 
                              columns=df.columns)
            if len(df.columns) != orig_n_vars:
                raise RuntimeError('Mean imputation failed.' + \
                    'Please select another imputation strategy.')
        elif self._imputation_strategy == 'drop':
            orig_len = len(df)
            df = df.dropna(axis=0)
            if len(df) < orig_len / 2:
                print(f'{orig_len - len(df)} entries ' + \
                      f'(over half of all {orig_len} entries) dropped.' + \
                    ' Please consider selecting a subset of features before' + \
                    ' handling missing data.')
        elif self._imputation_strategy is None:
            pass
        else:
            raise ValueError(f'Invalid imputation strategy.')

        return df





