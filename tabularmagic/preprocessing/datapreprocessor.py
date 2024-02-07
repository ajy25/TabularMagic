import pandas as pd
import numpy as np
from typing import Literal
from sklearn.impute import KNNImputer, SimpleImputer


class BaseSingleVarScaler():

    def __init__(self, var_name: str, x: np.ndarray):
        pass

    def fit(self):
        pass

    def transform(self, x: np.ndarray):
        pass

    def inverse_transform(self, x_scaled: np.ndarray):
        pass

    def fit_transform(self, x: np.ndarray = None):
        if x is not None:
            self.x = x[~np.isnan(x)]
        self.fit(self)
        self.transform(x)


class MinMaxSingleVar(BaseSingleVarScaler):
    """Min max scaling of a single variable"""

    def __init__(self, var_name: str, x: np.ndarray):
        self.var_name = var_name
        self.x = x[~np.isnan(x)]
        self.fit()

    def fit(self):
        self.min = self.x.min()
        self.max = self.x.max()

    def transform(self, x: np.ndarray):
        return (x - self.min) / (self.max - self.min)

    def inverse_transform(self, x_scaled: np.ndarray):
        return (self.max - self.min) * x_scaled + self.min
    


class StandardizeSingleVar(BaseSingleVarScaler):
    """Standard scaling of a single variable"""

    def __init__(self, var_name: str, x: np.ndarray):
        self.var_name = var_name
        self.x = x[~np.isnan(x)]
        self.fit()

    def fit(self):
        self.sigma = self.x.std()
        self.mu = self.x.mean()

    def transform(self, x: np.ndarray):
        return (x - self.mu) / self.sigma

    def inverse_transform(self, x_scaled: np.ndarray):
        return self.sigma * x_scaled + self.mu



class DataPreprocessor():
    """Automatic handling of one-hot-encoding, scaling, feature selection, 
    and missing data imputation. 

    The original order of the columns are not guaranteed to be maintained. 
    """


    def __init__(self, df: pd.DataFrame,
                 onehot_vars: list[str] = [],
                 standardize_vars: list[str] = [],
                 minmax_vars: list[str] = [], 
                 imputation_strategy: Literal[None, 'drop', 'mean', 
                        'median', '5nn', '10nn'] = None,
                 dropfirst_onehot: bool = False):
        """Initializes a DataPreprocessor object. 

        Parameters
        ----------
        - df : pd.DataFrame
        - onehot_vars : list[str]. 
        - standard_scale_vars : list[str].
        - minmax_scale_vars : list[str].
        - imputation_strategy: Literal[None, 'drop', 'mean', 
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
        # print('begin forward')
        df_temp = df.copy()
        # print('finished copy')
        df_temp = self._onehot_forward(df_temp)
        # print('finished onehot')
        df_temp = self._standardize_forward(df_temp)
        # print('finished standardize')
        df_temp = self._minmax_forward(df_temp)
        # print('finished minmax')
        df_temp = self._handle_missing_values(df_temp)
        # print('finished missing')
        return df_temp
    

    def backward(self,  df: pd.DataFrame) -> pd.DataFrame:
        """Backward transformation on an entire DataFrame. 
        
        Parameters
        ----------
        - df : pd.DataFrame.

        Returns
        -------
        - pd.DataFrame. 
        """
        df_temp = df.copy()
        df_temp = self._minmax_backward(df_temp)
        df_temp = self._standardize_backward(df_temp)
        df_temp = self._onehot_backward(df_temp)
        return df_temp
    

    def forward_single_var(self, series: pd.Series) -> pd.Series:
        """Forward transformation on a single Series. Only for continuous 
        variable transformations. 
        
        Parameters
        ----------
        - series : pd.Series.

        Returns
        -------
        - pd.Series. 
        """
        var_name = str(series.name)
        if var_name in self._minmax_scale_vars:
            return pd.Series(data=self._minmax_scale_var_to_scaler[var_name].\
                transform(series.to_numpy()), index=series.index)
        elif var_name in self._standard_scale_vars:
            return pd.Series(data=self._standard_scale_var_to_scaler[var_name].\
                transform(series.to_numpy()), index=series.index)
        else:
            raise ValueError(f'Invalid input: {series}. Name of series must' + \
                ' be in standardized/min-maxed variable list.')
        

    def backward_single_var(self, series: pd.Series) -> pd.Series:
        """Backward transformation on a single Series. Only for continuous 
        variable transformations. 
        
        Parameters
        ----------
        - series : pd.Series.

        Returns
        -------
        - pd.Series. 
        """
        var_name = str(series.name)
        if var_name in self._minmax_scale_vars:
            return pd.Series(data=self._minmax_scale_var_to_scaler[var_name].\
                inverse_transform(series.to_numpy()), index=series.index)
        elif var_name in self._standard_scale_vars:
            return pd.Series(data=self._standard_scale_var_to_scaler[var_name].\
                inverse_transform(series.to_numpy()), index=series.index)
        else:
            raise ValueError(f'Invalid input: {series}. Name of series must' + \
                ' be in standardized/min-maxed variable list.')
        

    def get_single_var_scaler(self, var_name: str) -> BaseSingleVarScaler:
        if var_name in self._minmax_scale_vars:
            return self._minmax_scale_var_to_scaler[var_name]
        elif var_name in self._standard_scale_vars:
            return self._standard_scale_var_to_scaler[var_name]
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
            df = pd.concat((df, temp_encoded), axis=1)
        self._transformed_vars = df.columns.to_list()
        return df
    

    def _onehot_backward(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse one hot encodes subset of DataFrame. 
        If the variable is binary, then nothing will be done. 
        
        Parameters
        ----------
        - df : pd.DataFrame.

        Returns
        -------
        - pd.DataFrame.
        """
        for var in self._onehot_vars:
            encoded_vars = self._onehot_var_to_encoded[var]
            temp_decoded = df[encoded_vars].idxmax(axis=1)
            df.drop(columns=encoded_vars)
            df = pd.concat((df, temp_decoded))
        return df[self.orig_df.columns]


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
    
    def _standardize_backward(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse standardizes subset of the DataFrame. 
        
        Parameters 
        ----------
        - df : pd.DataFrame

        Returns
        -------
        - pd.DataFrame
        """
        for var in self._standard_scale_vars:
            df[var] = self._standard_scale_var_to_scaler[var].inverse_transform(
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
    
    def _minmax_backward(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse min max scales subset of the DataFrame. 
        
        Parameters 
        ----------
        - df : pd.DataFrame.

        Returns
        -------
        - pd.DataFrame.
        """
        for var in self._minmax_scale_vars:
            df[var] = self._minmax_scale_var_to_scaler[var].inverse_transform(
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
                raise RuntimeWarning('Over half of all entries dropped.' + \
                    ' Please consider selecting a subset of features before' + \
                    ' handling missing data.')
        elif self._imputation_strategy is None:
            pass
        else:
            raise ValueError(f'Invalid imputation strategy.')
        return df





