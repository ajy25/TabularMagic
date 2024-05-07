import pandas as pd
import numpy as np
from typing import Literal, Callable
from sklearn.impute import KNNImputer, SimpleImputer
from ..util.console import print_wrapped



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





class DataPreprocessor:
    """Automatic handling of one-hot-encoding, scaling, feature selection, 
    and missing data imputation. 

    The original order of the columns are not guaranteed to be maintained. 
    """


    def __init__(self, df: pd.DataFrame,
                 imputation_strategy: Literal[None, 'drop', 
                    'median-mostfrequent', 'mean-mostfrequent', 
                    '5nn-mostfrequent'] = None,
                 onehot_vars: list[str] = [],
                 standardize_vars: list[str] = [],
                 minmax_vars: list[str] = [], 
                 log1p_vars: list[str] = [],
                 log_vars: list[str] = [],
                 dropfirst_onehot: bool = False, 
                 verbose: bool = False):
        """Initializes a DataPreprocessor object. 

        Parameters
        ----------
        - df : pd.DataFrame.
        - imputation_strategy : Literal[None, 'drop', 
            'median-mostfrequent', 'mean-mostfrequent', 
            '5nn-mostfrequent']. 
            Imputation strategy described for 
            continuous-categorical variables. 
        - onehot_vars : list[str]. 
        - standard_scale_vars : list[str].
        - minmax_scale_vars : list[str].
        - log1p_vars : list[str].
        - log_vars : list[str]
        - dropfirst_onehot : bool. 
        - verbose : bool.
        
        Returns
        -------
        - pd.DataFrame
        """
        self.orig_df = df.copy()
        self._verbose = verbose

        # missing data metadata
        self._imputation_strategy = imputation_strategy
        self._categorical_imputer = None
        self._continuous_imputer = None

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
        df_temp = self._handle_missing_values(df_temp)
        df_temp = self._standardize_forward(df_temp)
        df_temp = self._minmax_forward(df_temp)
        df_temp = self._log1p_forward(df_temp)
        df_temp = self._log_forward(df_temp)
        df_temp = self._onehot_forward(df_temp)
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

        orig_var_order = df.columns.to_list()

        categorical_vars = df.select_dtypes(
            include=['object', 'category', 'bool']).columns.to_list()
        continuous_vars = df.select_dtypes(
            exclude=['object', 'category', 'bool']).columns.to_list()


        if self._imputation_strategy == 'drop':
            orig_len = len(df)
            df_out = df.dropna(axis=0)
            if self._verbose:
                if len(df_out) < orig_len / 2:
                    print_wrapped(
                        f'{orig_len - len(df_out)} entries ' + \
                        f'(over half of all {orig_len} entries) dropped.' + \
                        ' Please consider selecting a subset of features ' + \
                        'before handling missing data', 
                        type='WARNING'
                    )
                else:
                    print_wrapped(
                        f'Dropped {orig_len - len(df_out)} entries',
                        type='UPDATE'
                    )
                    

        elif self._imputation_strategy in \
            ['median-mostfrequent', 'mean-mostfrequent', '5nn-mostfrequent']:

            orig_n_vars = len(df.columns)

            df_categorical = None
            df_continuous = None

            if len(categorical_vars) > 0:
                self._categorical_imputer = SimpleImputer(
                    strategy='most_frequent',
                    keep_empty_features=True)
                df_categorical = pd.DataFrame(
                    self._categorical_imputer.fit_transform(
                        df[categorical_vars]), 
                    columns=categorical_vars)
                
            if len(continuous_vars) > 0:
                
                if self._imputation_strategy == 'median-mostfrequent':
                    self._continuous_imputer = SimpleImputer(
                        strategy='median', 
                        keep_empty_features=True)
                elif self._imputation_strategy == 'mean-mostfrequent':
                    self._continuous_imputer = SimpleImputer(
                        strategy='mean', 
                        keep_empty_features=True)
                elif self._imputation_strategy == '5nn-mostfrequent':
                    self._continuous_imputer = KNNImputer(
                        n_neighbors=5, 
                        keep_empty_features=True)
                    
                df_continuous = pd.DataFrame(
                    self._continuous_imputer.fit_transform(
                        df[continuous_vars]), 
                    columns=continuous_vars)
                
            
            if df_categorical is not None and df_continuous is None:
                df_out = df_categorical
            elif df_continuous is not None and df_categorical is None:
                df_out = df_continuous
            else:
                df_out = df_continuous.join(df_categorical)[orig_var_order]

            if len(df_out.columns) != orig_n_vars:
                raise RuntimeError(
                    f'{self._imputation_strategy} imputation failed.' + \
                    'Please select another imputation strategy.')
            

        elif self._imputation_strategy is None:
            pass
        else:
            raise ValueError(f'Invalid imputation strategy.')

        return df_out




