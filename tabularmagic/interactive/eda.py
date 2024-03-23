import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew, pearsonr
from typing import Literal, Iterable
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

    def plot_distribution(self, figsize: Iterable = (5, 5), 
                          density: bool = False):
        """Returns a figure that is a bar plot of the relative frequencies
        of the data.
        
        Parameters 
        ----------
        - figsize : Iterable
        - density : bool

        Returns
        -------
        - plt.Figure
        """
        value_freqs = self._var_series.value_counts(normalize=density)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.bar(value_freqs.index, value_freqs.values, color='black', 
               edgecolor='black')
        ax.set_title(f'Distrubution of {self.variable_name}')
        ax.set_xlabel('Categories')
        if density:
            ax.set_ylabel('Density')
        else:
            ax.set_ylabel('Frequency')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
        fig.tight_layout()
        plt.close(fig)
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
            'n_missing': self._var_series.isna().sum(),
            'n': len(self._var_series)
        }
        self.summary_statistics = pd.DataFrame(
            list(self._summary_statistics_dict.items()), 
            columns=['Statistic', self.variable_name]
        ).set_index('Statistic')

    def plot_distribution(self, figsize: Iterable = (5, 5),
            hypothetical_transform: \
                Literal[None, 'minmax', 'standardize', 'log1p'] = None,
            density: bool = False):
        """Returns a figure that is a histogram.
        
        Parameters 
        ----------
        - figsize : Iterable
        - hypothetical_transform : Literal[None, 'minmax', 
            'standardize', 'log1p']
            Default: None. 
        - density : bool.
            
        Returns
        -------
        - plt.Figure
        """
        values = self._var_series.to_numpy()

        if density:
            stat = 'density'
        else:
            stat = 'count'

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

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        sns.histplot(values, bins='auto', color='black', edgecolor='black', 
                     stat=stat, ax=ax, kde=True)
        ax.set_title(f'Distribution of {self.variable_name}')
        ax.set_xlabel('Values')
        if density:
            ax.set_ylabel('Density')
        else:
            ax.set_ylabel('Frequency')
        ax.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3))
        fig.tight_layout()
        plt.close(fig)
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
        self.df = df.copy()
        self.categorical_columns = df.select_dtypes(
            include=['object', 'category', 'bool']).columns.to_list()
        self.continuous_columns = df.select_dtypes(
            exclude=['object', 'category', 'bool']).columns.to_list()
        self._categorical_eda_dict = {
            var: CategoricalEDA(self.df[var]) \
                for var in self.categorical_columns
        }
        self._continuous_eda_dict = {
            var: ContinuousEDA(self.df[var]) \
                for var in self.continuous_columns
        }
        if len(self.categorical_columns) > 0:
            self.categorical_summary_statistics = pd.concat(
                [eda.summary_statistics\
                for eda in self._categorical_eda_dict.values()], axis=1)
        if len(self.continuous_columns) > 0:
            self.continuous_summary_statistics = pd.concat(
                [eda.summary_statistics\
                for eda in self._continuous_eda_dict.values()], axis=1) 
            

    def plot_continuous_pairs(self, continuous_vars: list[str] = None, 
                                    stratify_by_var: str = None, 
                                    figsize: Iterable = (5, 5)):
        """
        Plots pairwise relationships between continuous variables. 

        Parameters
        ----------
        - continuous_vars : list[str]. 
            A list of continuous variables. Default: None. 
            If None, all continuous variables are considered.
        - stratify_by_var : str.
            Categorical var name. 

        Returns
        -------
        - plt.Figure
        """
        if continuous_vars is None:
            continuous_vars = self.continuous_columns
        if len(continuous_vars) > 6:
            raise ValueError('No more than 6 continuous variables may be ' + \
                             'plotted at the same time.')
        if stratify_by_var is None: 
            grid = sns.PairGrid(self.df[continuous_vars].dropna())
            right_adjust = None
        else:
            grid = sns.PairGrid(
                self.df[continuous_vars + [stratify_by_var]].dropna(), 
                hue=stratify_by_var)
            right_adjust = 0.85


        grid.map_diag(sns.histplot, color='black', edgecolor=None, kde=True)
        if stratify_by_var is None:
            grid.map_upper(lambda x, y, **kwargs: plt.text(0.5, 0.5,
                f'ρ = {round(pearsonr(x, y)[0], 3)}\n' + \
                f'p = {round(pearsonr(x, y)[1], 5)}', 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=8))
        else:
            grid.map_upper(sns.scatterplot, s=2, color='black')
        grid.map_lower(sns.scatterplot, s=2, color='black')

        grid.add_legend()
        for ax in grid.axes.flat:
            ax.tick_params(axis='both', which='both', 
                           labelsize=5)
            ax.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3))
            ax.xaxis.offsetText.set_fontsize(5)
            ax.yaxis.offsetText.set_fontsize(5)
            ax.set_xlabel(ax.get_xlabel(), fontsize=6)
            ax.set_ylabel(ax.get_ylabel(), fontsize=6)

        fig = grid.figure
        fig.set_size_inches(*figsize)
        fig.subplots_adjust(wspace=0.2, hspace=0.2, right=right_adjust)

        if stratify_by_var is not None: 
            legend = fig.legends[0]
            legend.set_title(legend.get_title().get_text(), \
                             prop={'size': 7})
            # TODO: Fix label spacing
            for text in legend.get_texts():
                text.set_fontsize(6)
        plt.close(fig)
        return fig
    

    def mean_test(self, continuous_var: str, stratify_by_var: str):
        """Conducts the appropriate statistical test. 

        Null hypothesis: the means of all groups are the same.

        Parameters
        ----------
        - continuous_var : str. 
            Continuous var name to be stratified and compared. 
        - stratify_by_var : str.
            Categorical var name. 

        Returns
        -------
        - 
        """
        if continuous_var not in self.continuous_columns:
            raise ValueError(
                f'Invalid input: {continuous_var}. ' + \
                'Must be a known continuous variable.'
            )
        if stratify_by_var not in self.categorical_columns:
            raise ValueError(
                f'Invalid input: {stratify_by_var}. ' + \
                'Must be a known categorical variable.'
            )
        
        

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







