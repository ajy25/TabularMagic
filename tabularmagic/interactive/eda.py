import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from typing import Literal
from sklearn.preprocessing import minmax_scale, scale


class CategoricalEDA():
    """CategoricalEDA: generates EDA-relevant plots and tables for a 
    single categorical variable."""

    def __init__(self, var_series: pd.Series):
        """
        Initializes a CategoricalEDA object.

        Parameters
        ----------
        - var_series : pd.Series

        Returns
        -------
        - None
        """
        self.variable_name = str(var_series.name)
        self._var_series = var_series
        self._summary_statistics_dict = {
            'n_unique_vals': self._var_series.nunique(),
            'most_common_val': self._var_series.\
                value_counts(dropna=True).idxmax(),
            'least_common_val': self._var_series.\
                value_counts(dropna=True).idxmin(), 
            'n_missing_samples': self._var_series.isna().sum(),
            'n_samples': len(self._var_series)
        }
        self.summary_statistics = pd.DataFrame(
            list(self._summary_statistics_dict.items()), 
            columns=['Statistic', self.variable_name]
        ).set_index('Statistic')

    def plot_distribution(self):
        """Returns a figure that is a bar plot of the relative frequencies
        of the data.
        
        Parameters 
        ----------
        - None

        Returns
        -------
        - plt.Figure
        """
        value_freqs = self._var_series.value_counts(normalize=True)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.bar(value_freqs.index, value_freqs.values, color='black')
        ax.set_title(f'Distrubution of {self.variable_name}')
        ax.set_xlabel('Categories')
        ax.set_ylabel('Density')
        fig.tight_layout()
        return fig


class ContinuousEDA():
    """ContinuousEDA: generates EDA-relevant plots and tables for a 
    single continuous-valued variable."""

    def __init__(self, var_series: pd.Series):
        """
        Initializes a ContinuousEDA object.

        Parameters
        ----------
        - var_series : pd.Series

        Returns
        -------
        - None
        """
        self.variable_name = str(var_series.name)
        self._var_series = var_series
        self._summary_statistics_dict = {
            'min': self._var_series.min(),
            'max': self._var_series.max(),
            'mean': self._var_series.mean(),
            'variance': self._var_series.var(),
            'skew': skew(self._var_series.dropna().to_numpy()),
            'kurtosis': kurtosis(self._var_series.dropna().to_numpy()),
            'q1': self._var_series.quantile(q=0.25),
            'median': self._var_series.median(),
            'q3': self._var_series.quantile(q=0.75),
            'n_missing_samples': self._var_series.isna().sum(),
            'n_samples': len(self._var_series)
        }
        self.summary_statistics = pd.DataFrame(
            list(self._summary_statistics_dict.items()), 
            columns=['Statistic', self.variable_name]
        ).set_index('Statistic')

    def plot_distribution(self, hypothetical_transform: Literal[None, 'minmax', 
            'standardize', 'log1p'] = None):
        """Returns a figure that is a histogram.
        
        Parameters 
        ----------
        - hypothetical_transform : Literal[None, 'minmax', 
            'standardize', 'log1p']
            Default: None. 
        Returns
        -------
        - plt.Figure
        """
        values = self._var_series.to_numpy()

        if hypothetical_transform is None:
            pass
        elif hypothetical_transform == 'minmax':
            values = minmax_scale(values)
        elif hypothetical_transform == 'standardize':
            values = scale(values)
        elif hypothetical_transform == 'log1p':
            values = np.log1p(values)
        else:
            raise ValueError(f'Invalid input: {hypothetical_transform}.')

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.hist(values, bins='auto', color='black', 
                 edgecolor='black', density=True)
        ax.set_title(f'Distribution of {self.variable_name}')
        ax.set_xlabel('Values')
        ax.set_ylabel('Density')
        fig.tight_layout()
        return fig


class ComprehensiveEDA():
    """ComprehensiveEDA: generates EDA-relevant plots and tables for all 
    variables.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes a ComprehensiveEDA object. 

        Parameters 
        ----------
        - df : pd.DataFrame 

        Returns
        -------
        - None
        """
        self.df = df
        self.categorical_columns = df.select_dtypes(
            include=['object', 'category']).columns
        self.continuous_columns = df.select_dtypes(
            exclude=['object', 'category']).columns
        self._categorical_eda_dict = {
            var: CategoricalEDA(self.df[var]) \
                for var in self.categorical_columns
        }
        self._continuous_eda_dict = {
            var: ContinuousEDA(self.df[var]) \
                for var in self.continuous_columns
        }
        if len(self.categorical_columns) > 0:
            self.categorical_summary_statistics = pd.concat([eda.summary_statistics\
                for eda in self._categorical_eda_dict.values()], axis=1)
        if len(self.continuous_columns) > 0:
            self.continuous_summary_statistics = pd.concat([eda.summary_statistics\
                for eda in self._continuous_eda_dict.values()], axis=1) 

    def __getitem__(self, index: str) -> CategoricalEDA | ContinuousEDA:
        """Indexes into ComprehensiveRegressionReport. 

        Parameters
        ----------
        - index : str. 

        Returns
        -------
        - CategoricalEDA | ContinuousEDA. 
        """
        if isinstance(index, str):
            if index in self.categorical_columns:
                return self._categorical_eda_dict[index]
            elif index in self.continuous_columns:
                return self._continuous_eda_dict[index]
            else:
                raise ValueError(f'Invalid input: {index}. Index must be a ' + \
                                 'variable name in the input df.')
        else:
            raise ValueError(f'Invalid input: {index}. Index must be a string.')


