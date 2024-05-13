import matplotlib.pyplot as plt
import matplotlib.axes as axes
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew, pearsonr, ttest_ind
from typing import Literal, Iterable
from sklearn.preprocessing import minmax_scale, scale
from sklearn.decomposition import PCA
from textwrap import fill


from .stattests import StatisticalTestResult



class CategoricalEDA:
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

        n_missing = self._var_series.isna().sum()

        self._summary_statistics_dict = {
            'n_unique': self._var_series.nunique(),
            'most_common': self._var_series.\
                value_counts(dropna=True).idxmax(),
            'least_common': self._var_series.\
                value_counts(dropna=True).idxmin(), 
            'n_missing': n_missing,
            'missing_rate': n_missing / len(self._var_series),
            'n': len(self._var_series)
        }
        self.summary_statistics = pd.DataFrame(
            list(self._summary_statistics_dict.items()), 
            columns=['Statistic', self.variable_name]
        ).set_index('Statistic')

    def plot_distribution(self, density: bool = False, 
                          figsize: Iterable = (5, 5), 
                          ax: axes.Axes = None):
        """Returns a figure that is a bar plot of the relative frequencies
        of the data.
        
        Parameters 
        ----------
        - density : bool
        - figsize : Iterable
        - ax : 

        Returns
        -------
        - plt.Figure
        """
        value_freqs = self._var_series.value_counts(normalize=density)

        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)


        ax.bar(value_freqs.index, value_freqs.values, color='gray', 
               edgecolor='none')
        ax.set_title(f'Distrubution of {self.variable_name}')
        ax.set_xlabel('Categories')
        if density:
            ax.set_ylabel('Density')
        else:
            ax.set_ylabel('Frequency')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))

        if fig is not None:
            fig.tight_layout()
            plt.close()
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

        n_missing = self._var_series.isna().sum()

        self._summary_statistics_dict = {
            'min': self._var_series.min(),
            'max': self._var_series.max(),
            'mean': self._var_series.mean(),
            'std': self._var_series.std(),
            'variance': self._var_series.var(),
            'skew': skew(self._var_series.dropna().to_numpy()),
            'kurtosis': kurtosis(self._var_series.dropna().to_numpy()),
            'q1': self._var_series.quantile(q=0.25),
            'median': self._var_series.median(),
            'q3': self._var_series.quantile(q=0.75),
            'n_missing': n_missing,
            'missing_rate': n_missing / len(self._var_series),
            'n': len(self._var_series)
        }
        self.summary_statistics = pd.DataFrame(
            list(self._summary_statistics_dict.items()), 
            columns=['Statistic', self.variable_name]
        ).set_index('Statistic')


    def plot_distribution(self,
            hypothetical_transform: \
                Literal[None, 'minmax', 'standardize', 'log', 'log1p'] = None,
            density: bool = False, 
            figsize: Iterable = (5, 5), 
            ax: axes.Axes = None):
        """Returns a figure that is a histogram.
        
        Parameters 
        ----------
        - hypothetical_transform : Literal[None, 'minmax', 
            'standardize', 'log1p']
            Default: None. 
        - density : bool.
        - figsize : Iterable.
        - ax : axes.Axes.
            
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
        elif hypothetical_transform == 'log':
            values = np.log(values)
        else:
            raise ValueError(f'Invalid input: {hypothetical_transform}.')

        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        sns.histplot(values, bins='auto', color='black', edgecolor='none', 
                     stat=stat, ax=ax, kde=True, alpha=0.2)
        ax.set_title(f'Distribution of {self.variable_name}')
        ax.set_xlabel('Values')
        if density:
            ax.set_ylabel('Density')
        else:
            ax.set_ylabel('Frequency')
        ax.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3))

        if fig is not None:
            fig.tight_layout()
            plt.close()
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
        self._categorical_vars = df.select_dtypes(
            include=['object', 'category', 'bool']).columns.to_list()
        self._continuous_vars = df.select_dtypes(
            exclude=['object', 'category', 'bool']).columns.to_list()
        self._categorical_eda_dict = {
            var: CategoricalEDA(self.df[var]) \
                for var in self._categorical_vars
        }
        self._continuous_eda_dict = {
            var: ContinuousEDA(self.df[var]) \
                for var in self._continuous_vars
        }
        if len(self._categorical_vars) > 0:
            self._categorical_summary_statistics = pd.concat(
                [eda.summary_statistics\
                for eda in self._categorical_eda_dict.values()], axis=1)
        if len(self._continuous_vars) > 0:
            self._continuous_summary_statistics = pd.concat(
                [eda.summary_statistics\
                for eda in self._continuous_eda_dict.values()], axis=1) 
            


    # --------------------------------------------------------------------------
    # PLOTTING
    # --------------------------------------------------------------------------


    def plot_continuous_pairs(self, continuous_vars: list[str] = None, 
        stratify_by: str = None, figsize: Iterable = (5, 5)):
        """
        Plots pairwise relationships between continuous variables. 

        Parameters
        ----------
        - continuous_vars : list[str]. 
            A list of continuous variables. Default: None. 
            If None, all continuous variables are considered.
        - stratify_by : str.
            Categorical var name. 
        - figsize : Iterable.

        Returns
        -------
        - plt.Figure
        """
        if continuous_vars is None:
            continuous_vars = self._continuous_vars
        if len(continuous_vars) > 6:
            raise ValueError('No more than 6 continuous variables may be ' + \
                             'plotted at the same time.')
        
        if stratify_by is None: 
            grid = sns.PairGrid(self.df[continuous_vars].dropna())
            right_adjust = None
        else:
            grid = sns.PairGrid(
                self.df[continuous_vars + [stratify_by]].dropna(), 
                hue=stratify_by, palette=sns.color_palette()\
                    [:len(self.df[stratify_by].unique())])
            right_adjust = 0.85

        grid.map_diag(sns.histplot, color='black', edgecolor='none', kde=True, 
                      alpha=0.2)
        
        if stratify_by is None:
            grid.map_upper(lambda x, y, **kwargs: plt.text(0.5, 0.5,
                f'ρ = {round(pearsonr(x, y)[0], 3)}\n' + \
                f'p = {round(pearsonr(x, y)[1], 5)}', 
                ha='center', va='center', 
                transform=plt.gca().transAxes,
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

        if stratify_by is not None: 
            legend = fig.legends[0]
            legend.set_title(legend.get_title().get_text(), \
                             prop={'size': 7})
            # TODO: Fix label spacing
            for text in legend.get_texts():
                text.set_text(fill(text.get_text(), 10))
                text.set_fontsize(6)
        fig.subplots_adjust(left=0.1, bottom=0.1)
        plt.close(fig)
        return fig
    


    def plot_pairs(self, vars: list[str] = None):
        """
        Plots pairwise relationships between variables. 

        Parameters
        ----------
        - vars : list[str]. 
            A list of continuous variables. Default: None. 
            If None, all continuous variables are considered.
        - figsize : Iterable.

        Returns
        -------
        - plt.Figure
        """



    def plot_distribution_stratified(self, continuous_var: str, 
        stratify_by: str, include_hist: bool = False, 
        figsize : Iterable = (5, 5), 
        ax: axes.Axes = None):
        """Plots the distributions (density) of a given continuous variable 
        stratified by a categorical variable. Note that NaNs will be dropped, 
        which may yield different ranges for different 
        stratify_by inputs, depending on their levels of missingness. 
        
        Parameters
        ----------
        - continuous_var : str. Continuous variable of interest.
        - stratify_by : str. 
        - include_hist: bool. If True, includes the histograms in addition to 
            the KDE plots. 
        - figsize : Iterable.
        - ax : axes.Axes. If not None, does not return a figure; plots the 
            plot directly onto the input axes.Axes. 

        Returns
        -------
        - plt.Figure
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        local_df = self.df[[continuous_var, stratify_by]].dropna()

        for i, category in enumerate(local_df[stratify_by].unique()):
            subset = local_df[local_df[stratify_by] == category]
            if include_hist:
                sns.histplot(subset[continuous_var], bins='auto', kde=True, 
                            label=str(category), alpha=0.2, stat='density', 
                            ax=ax, color=sns.color_palette()[i], 
                            edgecolor='none')
            else:
                sns.kdeplot(subset[continuous_var], label=str(category), ax=ax)

        legend = ax.legend()
        legend.set_title(stratify_by)
        ax.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3))
        ax.set_title(f'Distribution of {continuous_var}')

        if fig is not None:
            fig.tight_layout()
            plt.close()
        return fig
    


    def plot_distribution(self, var: str, density: bool = False,
            figsize: Iterable = (5, 5), ax: axes.Axes = None):
        """Plots the distribution of the variable.
        
        Parameters
        ----------
        - var : str.
        - density : bool. If True, plots the density rather than the 
            frequency.
        - figsize : Iterable.
        - ax : axes.Axes. If not None, does not return a figure; plots the 
            plot directly onto the input axes.Axes. 

        Returns
        -------
        - plt.Figure
        """
        return self[var].plot_distribution(
            density=density, figsize=figsize, ax=ax)




    def plot_pca(self, continuous_vars: list[str], stratify_by: str = None, 
                 strata: pd.Series = None, 
                 standardize: bool = True, whiten: bool = False,
                 three_components: bool = False, figsize: Iterable = (5, 5), 
                 ax: axes.Axes = None):
        """Plots the first two (or three) principle components, 
        optionally stratified by an additional variable. Drops examples 
        with missing values across the given variables of interest.
        
        Parameters
        ----------
        - continuous_vars : list[str]. List of continuous variables across 
            which the PCA will be performed
        - stratify_by : str. Categorical variable from which strata are 
            identified. 
        - strata : Iterable[str].
            Must be the same length as the dataset. Index must be compatible 
            with self.df. Overidden by stratify_by if both provided. 
        - standardize : bool. If True, centers and scales each feature to have 
            0 mean and unit variance. 
        - whiten : bool. If True, performs whitening on the data during PCA. 
        - three_components : If True, returns a 3D plot. Otherwise plots the 
            first two components only. 
        - figsize : Iterable.
        - ax : axes.Axes. If not None, does not return a figure; plots the 
            plot directly onto the input axes.Axes. 

        Returns
        -------
        - plt.Figure
        """
        if strata is not None:
            if len(strata) != len(self.df):
                raise ValueError('strata must have same length ' +\
                                 'as self.df.')
            elif stratify_by is not None:
                raise ValueError('One of stratify_by, strata' +\
                                 ' must be None.')
            else:
                pass

        fig = None
        if ax is None:
            if three_components:
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111, projection='3d')
            else:
                fig, ax = plt.subplots(1, 1, figsize=figsize)

        if three_components:
            pca = PCA(n_components=3, whiten=whiten)
        else:
            pca = PCA(n_components=2, whiten=whiten)

        if stratify_by is not None:
            X_y = self.df[continuous_vars].join(
                self.df[stratify_by]).dropna()
            X = X_y[continuous_vars].to_numpy()
            if standardize:
                X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            components = pca.fit_transform(X)
            categories = X_y[stratify_by].to_numpy()
            for category in np.unique(categories):
                mask = categories == category
                if three_components:
                    ax.scatter(components[mask, 0], components[mask, 1], 
                        components[mask, 2], label=category, s=2)
                    ax.set_xlabel('Principle Component 1')
                    ax.set_ylabel('Principle Component 2')
                    ax.set_zlabel('Principle Component 3')
                else:
                    ax.scatter(components[mask, 0], components[mask, 1], 
                        label=category, s=2)
                    ax.set_xlabel('Principle Component 1')
                    ax.set_ylabel('Principle Component 2')
            legend = ax.legend()
            legend.set_title(stratify_by)
        elif strata is not None:
            X_y = self.df[continuous_vars].join(strata).dropna()
            X = X_y[continuous_vars].to_numpy()
            if standardize:
                X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            components = pca.fit_transform(X)
            labels_name = strata.name
            categories = X_y[labels_name].to_numpy()
            for category in np.unique(categories):
                mask = categories == category
                if three_components:
                    ax.scatter(components[mask, 0], components[mask, 1], 
                        components[mask, 2], label=category, s=2)
                    ax.set_xlabel('Principle Component 1')
                    ax.set_ylabel('Principle Component 2')
                    ax.set_zlabel('Principle Component 3')
                else:
                    ax.scatter(components[mask, 0], components[mask, 1], 
                        label=category, s=2)
                    ax.set_xlabel('Principle Component 1')
                    ax.set_ylabel('Principle Component 2')
            legend = ax.legend()
            legend.set_title(labels_name)
        else:
            X = self.df[continuous_vars].dropna().to_numpy()
            if standardize:
                X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            components = pca.fit_transform(X)
            if three_components:
                ax.scatter(components[:, 0], components[:, 1], 
                        components[:, 2], color='black')
                ax.set_xlabel('Principle Component 1')
                ax.set_ylabel('Principle Component 2')
                ax.set_zlabel('Principle Component 3')
            else:
                ax.scatter(components[:, 0], components[:, 1], color='black', 
                    s=2)
                ax.set_xlabel('Principle Component 1')
                ax.set_ylabel('Principle Component 2')

        title_str = ', '.join(continuous_vars)
        ax.set_title(f'PCA({title_str})', wrap=True)
        ax.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3))
        
        if fig is not None:
            if not three_components:
                fig.tight_layout()
            else:
                fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            plt.close()
        return fig
    


    # --------------------------------------------------------------------------
    # TESTING
    # --------------------------------------------------------------------------
    def test_difference_in_means(self, continuous_var: str, 
                                 stratify_by: str):
        """Conducts the appropriate statistical test between two groups. 
        The parameter stratify_by must be the name of a binary variable. 

        Null hypothesis: mu_1 = mu_2.

        Parameters
        ----------
        - continuous_var : str. 
            Continuous var name to be stratified and compared. 
        - stratify_by : str.
            Categorical or continuous var name. Must be binary. 

        Returns
        -------
        - StatisticalTestResult
        """

        if continuous_var not in self._continuous_vars:
            raise ValueError(
                f'Invalid input: {continuous_var}. ' + \
                'Must be a known continuous variable.'
            )
        if (stratify_by not in self._categorical_vars) and \
            (stratify_by not in self._continuous_vars):
            raise ValueError(
                f'Invalid input: {stratify_by}. ' + \
                'Must be a known binary variable.'
            )
        
        categories = np.unique(self.df[stratify_by].to_numpy())
        if len(categories) != 2:
            raise ValueError(
                f'Invalid stratify_by: {stratify_by}. ' + \
                'Must be a known binary variable.'
            )
        
        group_1 =\
            self.df.loc[self.df[stratify_by] == categories[0], 
                        continuous_var].to_numpy()
        group_2 =\
            self.df.loc[self.df[stratify_by] == categories[1], 
                        continuous_var].to_numpy()

        ttest_result = ttest_ind(group_1, group_2, equal_var=False)


        return StatisticalTestResult(
            description='Welch\'s t-test',
            statistic=ttest_result.statistic,
            pval=ttest_result.pvalue,
            descriptive_statistic=float(group_1.mean() - group_2.mean()),
            degfree=ttest_result.df,
            statistic_description='t-statistic',
            descriptive_statistic_description='mu_1 - mu_2',
            null_hypothesis_description='mu_1 = mu_2',
            alternative_hypothesis_description='mu_1 != mu_2',
            long_description=f'Group 1 label: {categories[0]} \n' + \
                f'Gorup 2 label: {categories[1]}'
        )




    # --------------------------------------------------------------------------
    # GETTERS
    # --------------------------------------------------------------------------
    def continuous_vars(self):
        return self._continuous_vars
    
    def categorical_vars(self):
        return self._categorical_vars
    
    def categorical_summary_statistics(self):
        return self._categorical_summary_statistics
    
    def continuous_summary_statistics(self):
        return self._continuous_summary_statistics





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
            if index in self._categorical_vars:
                return self._categorical_eda_dict[index]
            elif index in self._continuous_vars:
                return self._continuous_eda_dict[index]
            else:
                raise ValueError(f'Invalid input: {index}. Index must be a ' + \
                                 'variable name in the input df.')
        else:
            raise ValueError(f'Invalid input: {index}. Index must be a string.')







