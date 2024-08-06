import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Literal
from sklearn.preprocessing import minmax_scale, scale
from sklearn.decomposition import PCA
from textwrap import fill
from .stattests import StatisticalTestResult
from ..display.print_utils import print_wrapped


class CategoricalEDA:
    """Class for generating EDA-relevant plots and tables for a
    single categorical variable.
    """

    def __init__(self, var_series: pd.Series):
        """
        Initializes a CategoricalEDA object.

        Parameters
        ----------
        var_series : pd.Series
            Pandas Series for a sample of the categorical variable.
        """
        self.variable_name = str(var_series.name)
        self._var_series = var_series

        n_missing = self._var_series.isna().sum()

        self._summary_statistics_dict = {
            "n_unique": self._var_series.nunique(),
            "most_common": self._var_series.value_counts(dropna=True).idxmax(),
            "least_common": self._var_series.value_counts(dropna=True).idxmin(),
            "n_missing": n_missing,
            "missing_rate": n_missing / len(self._var_series),
            "n": len(self._var_series),
        }
        self.summary_statistics = pd.DataFrame(
            list(self._summary_statistics_dict.items()),
            columns=["Statistic", self.variable_name],
        ).set_index("Statistic")

    def plot_distribution(
        self,
        density: bool = False,
        figsize: tuple[float, float] = (5, 5),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is a bar plot of the relative frequencies
        of the data.

        Parameters
        ----------
        density : bool
            Default: False. If True, plots the density rather than the
            frequency.

        figsize : tuple[float, float]
            Default: (5, 5). The size of the figure. Only used if
            ax is None.

        ax : plt.Axes | None
            Default: None. The axes to plot on. If None, a new figure is
            created.

        Returns
        -------
        plt.Figure
            Figure of the distribution.
        """
        value_freqs = self._var_series.value_counts(normalize=density)

        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax.bar(value_freqs.index, value_freqs.values, color="gray", edgecolor="none")
        ax.set_title(f"Distrubution of {self.variable_name}")
        ax.set_xlabel("Categories")
        if density:
            ax.set_ylabel("Density")
        else:
            ax.set_ylabel("Frequency")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))

        if fig is not None:
            fig.tight_layout()
            plt.close()
        return fig

    def counts(self) -> pd.Series:
        """Returns the counts of each category in the variable.

        Returns
        -------
        pd.Series
        """
        return self._var_series.value_counts(normalize=False)


class NumericEDA:
    """Class for generating EDA-relevant plots and tables for a
    single numeric variable."""

    def __init__(self, var_series: pd.Series):
        """
        Initializes a NumericEDA object.

        Parameters
        ----------
        var_series : pd.Series
            Pandas Series for a sample of the numeric variable.
        """
        self.variable_name = str(var_series.name)
        self._var_series = var_series

        n_missing = self._var_series.isna().sum()

        self._summary_statistics_dict = {
            "min": self._var_series.min(),
            "max": self._var_series.max(),
            "mean": self._var_series.mean(),
            "std": self._var_series.std(),
            "variance": self._var_series.var(),
            "skew": stats.skew(self._var_series.dropna().to_numpy()),
            "kurtosis": stats.kurtosis(self._var_series.dropna().to_numpy()),
            "q1": self._var_series.quantile(q=0.25),
            "median": self._var_series.median(),
            "q3": self._var_series.quantile(q=0.75),
            "n_missing": n_missing,
            "missing_rate": n_missing / len(self._var_series),
            "n": len(self._var_series),
        }
        self.summary_statistics = pd.DataFrame(
            list(self._summary_statistics_dict.items()),
            columns=["Statistic", self.variable_name],
        ).set_index("Statistic")

    def plot_distribution(
        self,
        hypothetical_transform: Literal[
            None, "minmax", "standardize", "log", "log1p"
        ] = None,
        density: bool = False,
        figsize: tuple[float, float] = (5, 5),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Returns a figure that is a histogram.

        Parameters
        ----------
        hypothetical_transform : Literal[None, 'minmax', 'standardize',
            'log', 'log1p'] | None
            Default: None. If not None, the data is transformed before
            plotting.

        density : bool
            Default: False. If True, plots the density rather than the
            frequency.

        figsize : tuple[float, float]
            Default: (5, 5). The size of the figure. Only used if
            ax is None.

        ax : plt.Axes | None
            Default: None. The axes to plot on. If None, a new figure is
            created.

        Returns
        -------
        plt.Figure
        """
        values = self._var_series.to_numpy()

        if density:
            stat = "density"
        else:
            stat = "count"

        if hypothetical_transform is None:
            pass
        elif hypothetical_transform == "minmax":
            values = minmax_scale(values)
        elif hypothetical_transform == "standardize":
            values = scale(values)
        elif hypothetical_transform == "log1p":
            values = np.log1p(values)
        elif hypothetical_transform == "log":
            values = np.log(values)
        else:
            raise ValueError(f"Invalid input: {hypothetical_transform}.")

        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        sns.histplot(
            values,
            bins="auto",
            color="black",
            edgecolor="none",
            stat=stat,
            ax=ax,
            kde=True,
            alpha=0.2,
        )
        ax.set_title(f"Distribution of {self.variable_name}")
        ax.set_xlabel("Values")
        if density:
            ax.set_ylabel("Density")
        else:
            ax.set_ylabel("Frequency")
        ax.ticklabel_format(style="sci", axis="both", scilimits=(-3, 3))

        if fig is not None:
            fig.tight_layout()
            plt.close()
        return fig


class EDAReport:
    """Class for generating EDA-relevant plots and tables for all
    variables.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes a EDAReport object.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset.
        """
        self._df = df.copy()
        self._categorical_vars = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.to_list()
        self._numeric_vars = df.select_dtypes(
            exclude=["object", "category", "bool"]
        ).columns.to_list()
        self._categorical_eda_dict = {
            var: CategoricalEDA(self._df[var]) for var in self._categorical_vars
        }
        self._numeric_eda_dict = {
            var: NumericEDA(self._df[var]) for var in self._numeric_vars
        }

        self._categorical_summary_statistics = None
        self._numeric_summary_statistics = None
        if len(self._categorical_vars) > 0:
            self._categorical_summary_statistics = pd.concat(
                [eda.summary_statistics for eda in self._categorical_eda_dict.values()],
                axis=1,
            )
        if len(self._numeric_vars) > 0:
            self._numeric_summary_statistics = pd.concat(
                [eda.summary_statistics for eda in self._numeric_eda_dict.values()],
                axis=1,
            )

    # --------------------------------------------------------------------------
    # PLOTTING
    # --------------------------------------------------------------------------

    def plot_numeric_pairs(
        self,
        numeric_vars: list[str] | None = None,
        stratify_by: str | None = None,
        figsize: tuple[float, float] = (7, 7),
    ) -> plt.Figure:
        """
        Plots pairwise relationships between numeric variables.

        Parameters
        ----------
        numeric_vars : list[str]
            Default: None. A list of numeric variables.
            If None, all numeric variables are considered.

        stratify_by : str
            Default: None. Categorical var name.
            If not None, the plot is stratified by this variable.

        figsize : tuple[float, float]
            Default: (7, 7). The size of the figure.

        Returns
        -------
        plt.Figure
        """
        if numeric_vars is None:
            numeric_vars = self._numeric_vars
        if len(numeric_vars) > 6:
            raise ValueError(
                "No more than 6 numeric variables may be " + "plotted at the same time."
            )

        if stratify_by is None:
            grid = sns.PairGrid(self._df[numeric_vars].dropna())
            right_adjust = None
        else:
            grid = sns.PairGrid(
                self._df[numeric_vars + [stratify_by]].dropna(),
                hue=stratify_by,
                palette=sns.color_palette()[: len(self._df[stratify_by].unique())],
            )
            right_adjust = 0.85

        grid.map_diag(
            sns.histplot, color="black", edgecolor="none", kde=True, alpha=0.2
        )

        if stratify_by is None:
            grid.map_upper(
                lambda x, y, **kwargs: plt.text(
                    0.5,
                    0.5,
                    f"ρ = {round(stats.pearsonr(x, y)[0], 3)}\n"
                    + f"p = {round(stats.pearsonr(x, y)[1], 5)}",
                    ha="center",
                    va="center",
                    transform=plt.gca().transAxes,
                    fontsize=8,
                )
            )
        else:
            grid.map_upper(sns.scatterplot, s=2, color="black")

        grid.map_lower(sns.scatterplot, s=2, color="black")

        grid.add_legend()
        for ax in grid.axes.flat:
            ax.tick_params(axis="both", which="both", labelsize=5)
            ax.ticklabel_format(style="sci", axis="both", scilimits=(-3, 3))
            ax.xaxis.offsetText.set_fontsize(5)
            ax.yaxis.offsetText.set_fontsize(5)
            ax.set_xlabel(ax.get_xlabel(), fontsize=6)
            ax.set_ylabel(ax.get_ylabel(), fontsize=6)

        fig = grid.figure
        fig.set_size_inches(*figsize)
        fig.subplots_adjust(wspace=0.2, hspace=0.2, right=right_adjust)

        if stratify_by is not None:
            legend = fig.legends[0]
            legend.set_title(legend.get_title().get_text(), prop={"size": 7})
            # TODO: Fix label spacing
            for text in legend.get_texts():
                text.set_text(fill(text.get_text(), 10))
                text.set_fontsize(6)
        fig.subplots_adjust(left=0.1, bottom=0.1)
        plt.close(fig)
        return fig

    def plot_distribution_stratified(
        self,
        numeric_var: str,
        stratify_by: str,
        strategy: Literal[
            "stacked_kde_density",
            "stacked_hist_kde_density",
            "stacked_hist_kde_frequency",
            "violin",
            "violin_swarm",
            "violin_strip",
            "box",
            "box_swarm",
            "box_strip",
        ] = "box",
        figsize: tuple[float, float] = (5, 5),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Plots the distributions (density) of a given numeric variable
        stratified by a categorical variable. Note that NaNs will be dropped,
        which may yield different ranges for different
        stratify_by inputs, depending on their levels of missingness.

        Parameters
        ----------
        numeric_var : str
            Numeric variable of interest.

        stratify_by : str
            Categorical variable to stratify by.

        strategy : Literal['stacked_kde_density', 'stacked_hist_kde_density',
            'stacked_hist_kde_frequency', 'violin', 'violin_swarm', 'violin_strip',
            'box', 'box_swarm', 'box_strip']
            Default: 'box'. The strategy for plotting the distribution.

        figsize : tuple[float, float]
            Default: (5, 5). The size of the figure.

        ax : plt.Axes | None
            Default: None. If not None, the plot is drawn on the input Axes.

        Returns
        -------
        plt.Figure
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        local_df = self._df[[numeric_var, stratify_by]].dropna()

        if strategy in [
            "stacked_hist_kde_frequency",
            "stacked_hist_kde_density",
            "stacked_kde_density",
        ]:
            for i, category in enumerate(local_df[stratify_by].unique()):
                subset = local_df[local_df[stratify_by] == category]
                if strategy == "stacked_hist_kde_density":
                    sns.histplot(
                        subset[numeric_var],
                        bins="auto",
                        kde=True,
                        label=str(category),
                        alpha=0.2,
                        stat="density",
                        ax=ax,
                        color=sns.color_palette()[i],
                        edgecolor="none",
                    )
                elif strategy == "stacked_hist_kde_frequency":
                    sns.histplot(
                        subset[numeric_var],
                        bins="auto",
                        kde=True,
                        label=str(category),
                        alpha=0.2,
                        stat="frequency",
                        ax=ax,
                        color=sns.color_palette()[i],
                        edgecolor="none",
                    )
                elif strategy == "stacked_kde_density":
                    sns.kdeplot(subset[numeric_var], label=str(category), ax=ax)

            legend = ax.legend()
            legend.set_title(stratify_by)

            ax.ticklabel_format(style="sci", axis="x", scilimits=(-3, 3))

        elif strategy == "violin_swarm":
            sns.violinplot(
                data=local_df,
                x=stratify_by,
                y=numeric_var,
                ax=ax,
                color="black",
                edgecolor="none",
                alpha=0.2,
            )
            sns.swarmplot(
                data=local_df,
                x=stratify_by,
                y=numeric_var,
                color="black",
                size=2,
                ax=ax,
            )

        elif strategy == "violin":
            sns.violinplot(
                data=local_df,
                x=stratify_by,
                y=numeric_var,
                ax=ax,
                color="black",
                edgecolor="none",
                alpha=0.2,
            )

        elif strategy == "violin_strip":
            sns.violinplot(
                data=local_df,
                x=stratify_by,
                y=numeric_var,
                ax=ax,
                color="black",
                edgecolor="none",
                alpha=0.2,
            )
            sns.stripplot(
                data=local_df,
                x=stratify_by,
                y=numeric_var,
                color="black",
                size=2,
                ax=ax,
            )

        elif strategy == "box_swarm":
            sns.boxplot(
                data=local_df,
                x=stratify_by,
                y=numeric_var,
                ax=ax,
                color="black",
                fill=False,
                linewidth=0.5,
            )
            sns.swarmplot(
                data=local_df,
                x=stratify_by,
                y=numeric_var,
                color="black",
                size=2,
                ax=ax,
            )

        elif strategy == "box_strip":
            sns.boxplot(
                data=local_df,
                x=stratify_by,
                y=numeric_var,
                ax=ax,
                color="black",
                fill=False,
                linewidth=0.5,
            )
            sns.stripplot(
                data=local_df,
                x=stratify_by,
                y=numeric_var,
                color="black",
                size=2,
                ax=ax,
            )

        elif strategy == "box":
            sns.boxplot(
                data=local_df,
                x=stratify_by,
                y=numeric_var,
                ax=ax,
                color="black",
                fill=False,
                linewidth=0.5,
            )

        else:
            raise ValueError(f"Invalid input: {strategy}.")

        ax.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))
        ax.set_title(f"Distribution of {numeric_var}")

        if fig is not None:
            fig.tight_layout()
            plt.close()
        return fig

    def plot_distribution(
        self,
        var: str,
        density: bool = False,
        figsize: tuple[float, float] = (5, 5),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Plots the distribution of the variable.

        Parameters
        ----------
        var : str
            Variable name.

        density : bool
            Default: False. If True, plots the density rather than the
            frequency.

        figsize : tuple[float, float]
            Default: (5, 5). The size of the figure. Only used if ax is None.

        ax : plt.Axes | None
            Default: None. If not None, the plot is drawn on the input Axes.

        Returns
        -------
        plt.Figure
        """
        return self[var].plot_distribution(density=density, figsize=figsize, ax=ax)

    def plot_pca(
        self,
        numeric_vars: list[str],
        stratify_by: str | None = None,
        strata: pd.Series | None = None,
        scale_strategy: Literal["standardize", "center", "none"] = "center",
        whiten: bool = False,
        three_components: bool = False,
        figsize: tuple[float, float] = (5, 5),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Plots the first two (or three) principle components,
        optionally stratified by an additional variable. Drops examples
        with missing values across the given variables of interest.

        Parameters
        ----------
        numeric_vars : list[str]
            List of numeric variables across which the PCA will be performed.

        stratify_by : str
            Categorical variable from which strata are identified.

        strata : pd.Series | None
            Default: None.
            The lables/strata.
            Must be the same length as the dataset. Index must be compatible
            with self.df. Overidden by stratify_by if both provided.

        scale_strategy : Literal["standardize", "center", "none"].
            Default: "center".

        whiten : bool
            Default: False. If True, performs whitening on the data during PCA.

        three_components : bool
            Default: False. If True, returns a 3D plot. Otherwise plots the
            first two components only.

        figsize : tuple[float, float]
            Default: (5, 5). The size of the figure. Only used if ax is None.

        ax : plt.Axes | None
            Default: None. If not None, does not return a figure; plots the
            plot directly onto the input Axes.

        Returns
        -------
        plt.Figure
        """
        if strata is not None:
            if len(strata) != len(self._df):
                raise ValueError("strata must have same length " + "as self.df.")
            elif stratify_by is not None:
                raise ValueError("One of stratify_by, strata" + " must be None.")
            else:
                pass

        fig = None
        if ax is None:
            if three_components:
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111, projection="3d")
            else:
                fig, ax = plt.subplots(1, 1, figsize=figsize)

        if three_components:
            pca = PCA(n_components=3, whiten=whiten)
        else:
            pca = PCA(n_components=2, whiten=whiten)

        if stratify_by is not None:
            X_y = self._df[numeric_vars].join(self._df[stratify_by]).dropna()
            X = X_y[numeric_vars].to_numpy()
            if scale_strategy == "standardize":
                X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            elif scale_strategy == "center":
                X = X - np.mean(X, axis=0)
            components = pca.fit_transform(X)
            categories = X_y[stratify_by].to_numpy()
            for category in np.unique(categories):
                mask = categories == category
                if three_components:
                    ax.scatter(
                        components[mask, 0],
                        components[mask, 1],
                        components[mask, 2],
                        label=category,
                        s=2,
                    )
                    ax.set_xlabel("Principle Component 1")
                    ax.set_ylabel("Principle Component 2")
                    ax.set_zlabel("Principle Component 3")
                else:
                    ax.scatter(
                        components[mask, 0], components[mask, 1], label=category, s=2
                    )
                    ax.set_xlabel("Principle Component 1")
                    ax.set_ylabel("Principle Component 2")
            legend = ax.legend()
            legend.set_title(stratify_by)
        elif strata is not None:
            X_y = self._df[numeric_vars].join(strata).dropna()
            X = X_y[numeric_vars].to_numpy()
            if scale_strategy == "standardize":
                X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            elif scale_strategy == "center":
                X = X - np.mean(X, axis=0)
            components = pca.fit_transform(X)
            labels_name = strata.name
            categories = X_y[labels_name].to_numpy()
            for category in np.unique(categories):
                mask = categories == category
                if three_components:
                    ax.scatter(
                        components[mask, 0],
                        components[mask, 1],
                        components[mask, 2],
                        label=category,
                        s=2,
                    )
                    ax.set_xlabel("Principle Component 1")
                    ax.set_ylabel("Principle Component 2")
                    ax.set_zlabel("Principle Component 3")
                else:
                    ax.scatter(
                        components[mask, 0], components[mask, 1], label=category, s=2
                    )
                    ax.set_xlabel("Principle Component 1")
                    ax.set_ylabel("Principle Component 2")
            legend = ax.legend()
            legend.set_title(labels_name)
        else:
            X = self._df[numeric_vars].dropna().to_numpy()
            if scale_strategy == "standardize":
                X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            elif scale_strategy == "center":
                X = X - np.mean(X, axis=0)
            components = pca.fit_transform(X)
            if three_components:
                ax.scatter(
                    components[:, 0], components[:, 1], components[:, 2], color="black"
                )
                ax.set_xlabel("Principle Component 1")
                ax.set_ylabel("Principle Component 2")
                ax.set_zlabel("Principle Component 3")
            else:
                ax.scatter(components[:, 0], components[:, 1], color="black", s=2)
                ax.set_xlabel("Principle Component 1")
                ax.set_ylabel("Principle Component 2")

        title_str = ", ".join(numeric_vars)
        ax.set_title(f"PCA({title_str})", wrap=True)
        ax.ticklabel_format(style="sci", axis="both", scilimits=(-3, 3))

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

    def test_equal_means(
        self, numeric_var: str, stratify_by: str
    ) -> StatisticalTestResult:
        """Conducts the appropriate statistical test to
        test for equal means between two ore more groups (null hypothesis).

        Parameters
        ----------
        numeric_var : str
            Numeric variable name to be stratified and compared.

        stratify_by : str
            Categorical variable name.

        Returns
        -------
        StatisticalTestResult
        """
        if stratify_by not in self._categorical_vars:
            raise ValueError(
                f"Invalid input: {stratify_by}. "
                "Must be a known categorical variable."
            )
        groups = self._df.groupby(stratify_by)[numeric_var].apply(list).to_dict()
        if len(groups) < 2:
            raise ValueError(
                "Invalid input: stratify_by. Must have at least two unique values."
            )
        elif len(groups) == 2:
            return self.ttest(numeric_var, stratify_by, "auto")
        else:
            return self.anova(numeric_var, stratify_by)

    def anova(
        self,
        numeric_var: str,
        stratify_by: str,
        strategy: Literal["auto", "anova_oneway", "kruskal"] = "auto",
    ) -> StatisticalTestResult:
        """Tests for equal means between three or more groups.
        Null hypothesis: All group means are equal.
        Alternative hypothesis: At least one group's mean is different from the others.
        NaNs in numeric_var and stratify_by are dropped before the test is conducted.


        Parameters
        ----------
        numeric_var : str
            Numeric variable name to be stratified and compared.

        stratify_by : str
            Categorical variable name.

        strategy : Literal['auto', 'anova_oneway', 'kruskal']
            Default: 'auto'. If 'auto', a test is selected as follows:
            If the data in any group is not normally distributed or not
            homoskedastic, then the Kruskal-Wallis test is used.
            Otherwise, the one-way ANOVA test is used. ANOVA is somewhat
            robust to heteroscedasticity and violations of the normality assumption.

        Returns
        -------
        StatisticalTestResult
        """
        if numeric_var not in self._numeric_vars:
            raise ValueError(
                f"Invalid input: {numeric_var}. " "Must be a known numeric variable."
            )
        if stratify_by not in self._categorical_vars:
            raise ValueError(
                f"Invalid input: {stratify_by}. "
                "Must be a known categorical variable."
            )

        local_df = self._df[[numeric_var, stratify_by]].dropna()

        categories = np.unique(local_df[stratify_by].to_numpy())

        if len(categories) < 3:
            raise ValueError(
                f"Invalid stratify_by: {stratify_by}. "
                "Must have at least three unique values."
            )

        groups = []
        for category in categories:
            groups.append(
                local_df.loc[local_df[stratify_by] == category, numeric_var].to_numpy()
            )

        auto_alpha = 0.05
        is_normal = True
        is_homoskedastic = stats.bartlett(*groups)[1] > auto_alpha

        for group in groups:
            if stats.shapiro(group)[1] <= auto_alpha:
                is_normal = False
                break

        if strategy == "auto":
            if is_normal and is_homoskedastic:
                strategy = "anova_oneway"
            else:
                strategy = "kruskal"

        if strategy == "kruskal":
            h_stat, p_val = stats.kruskal(*groups)
            return StatisticalTestResult(
                description="Kruskal-Wallis test",
                statistic=h_stat,
                pval=p_val,
                descriptive_statistic=None,
                degfree=None,
                statistic_description="H-statistic",
                descriptive_statistic_description=None,
                null_hypothesis_description="All group means are equal",
                alternative_hypothesis_description="At least one group's mean is "
                "different from the others",
            )

        elif strategy == "anova_oneway":
            f_stat, p_val = stats.f_oneway(*groups)

            return StatisticalTestResult(
                description="One-way ANOVA",
                statistic=f_stat,
                pval=p_val,
                descriptive_statistic=None,
                degfree=None,
                statistic_description="f-statistic",
                descriptive_statistic_description=None,
                null_hypothesis_description="All group means are equal",
                alternative_hypothesis_description="At least one group's mean is "
                "different from the others",
                assumptions_description="1. Data in each group are normally "
                "distributed. 2. Variances of each group are equal. "
                "3. Samples are independent.",
            )

    def ttest(
        self,
        numeric_var: str,
        stratify_by: str,
        strategy: Literal["auto", "student", "welch", "yuen", "mann-whitney"] = "welch",
    ) -> StatisticalTestResult:
        """Conducts the appropriate statistical test to test for equal means between
        two groups. The parameter stratify_by must be the name of a binary variable,
        i.e. a categorical or numeric variable with exactly two unique values.

        Null hypothesis: mu_1 = mu_2.
        Alternative hypothesis: mu_1 != mu_2
        This is a two-sided test.

        NaNs in numeric_var and stratify_by
            are dropped before the test is conducted.

        Parameters
        ----------
        numeric_var : str
            numeric variable name to be stratified and compared.

        stratify_by : str
            Categorical or numeric variable name. Must be binary.

        strategy : Literal['auto', 'student', 'welch', 'yuen', 'mann-whitney']
            Default: 'welch'.
            If 'auto', a test is selected as follows:
            If the data in either group is not normally distributed,
            and the variances are not equal, then Yuen's
            (20% trimmed mean) t-test is used.
            If the data in either group is not normally distributed,
            but the variances are equal, then the Mann-Whitney U test
            is used.
            If the data in both groups are normally distributed but the
            variances are not equal, Welch's t-test is used.
            Otherwise, Student's t-test is used.

        Returns
        -------
        StatisticalTestResult
        """

        if numeric_var not in self._numeric_vars:
            raise ValueError(
                f"Invalid input: {numeric_var}. " + "Must be a known numeric variable."
            )
        if (stratify_by not in self._categorical_vars) and (
            stratify_by not in self._numeric_vars
        ):
            raise ValueError(
                f"Invalid input: {stratify_by}. " + "Must be a known binary variable."
            )

        local_df = self._df[[numeric_var, stratify_by]].dropna()

        categories = np.unique(local_df[stratify_by].to_numpy())
        if len(categories) != 2:
            raise ValueError(
                f"Invalid stratify_by: {stratify_by}. "
                + "Must be a known binary variable."
            )

        group_1 = local_df.loc[
            local_df[stratify_by] == categories[0], numeric_var
        ].to_numpy()
        group_2 = local_df.loc[
            self._df[stratify_by] == categories[1], numeric_var
        ].to_numpy()

        if strategy == "auto":
            auto_alpha = 0.05

            try:
                normality1 = stats.shapiro(group_1)[1] > auto_alpha
                normality2 = stats.shapiro(group_2)[1] > auto_alpha
                is_normal = normality1 and normality2
            except Exception as e:
                print_wrapped(
                    f"Shapiro-Wilk test failed; assuming not normal: {e}.",
                    type="WARNING",
                )
                is_normal = False

            try:
                is_equal_var = stats.levene(group_1, group_2).pvalue > auto_alpha
            except Exception as e:
                print_wrapped(
                    f"Levene test failed; assuming unequal variances: {e}.",
                    type="WARNING",
                )
                is_equal_var = False

            if is_equal_var:
                if is_normal:
                    test_type = "student"
                else:
                    test_type = "welch"
            else:
                if is_normal:
                    test_type = "yuen"
                else:
                    test_type = "mann-whitney"

        elif strategy in ["student", "welch", "yuen", "mann-whitney"]:
            test_type = strategy

        else:
            raise ValueError(f"Invalid input: {strategy}.")

        if test_type == "student":
            ttest_result = stats.ttest_ind(
                group_1, group_2, equal_var=True, alternative="two-sided"
            )
            return StatisticalTestResult(
                description="Student's t-test",
                statistic=ttest_result.statistic,
                pval=ttest_result.pvalue,
                descriptive_statistic=float(group_1.mean() - group_2.mean()),
                degfree=ttest_result.df,
                statistic_description="t-statistic",
                descriptive_statistic_description="mu_1 - mu_2",
                null_hypothesis_description="mu_1 = mu_2",
                alternative_hypothesis_description="mu_1 != mu_2",
                assumptions_description="1. Var(Group 1) = Var(Group 2). "
                "2. Data in both groups are normally distributed.",
                long_description=f"Group 1 label: '{categories[0]}'. "
                f"Group 2 label: '{categories[1]}'. "
                "Variances of two groups are assumed to be equal.",
            )

        elif test_type == "welch":
            ttest_result = stats.ttest_ind(
                group_1, group_2, equal_var=False, alternative="two-sided"
            )
            return StatisticalTestResult(
                description="Welch's t-test",
                statistic=ttest_result.statistic,
                pval=ttest_result.pvalue,
                descriptive_statistic=float(group_1.mean() - group_2.mean()),
                degfree=ttest_result.df,
                statistic_description="t-statistic",
                descriptive_statistic_description="mu_1 - mu_2",
                null_hypothesis_description="mu_1 = mu_2",
                alternative_hypothesis_description="mu_1 != mu_2",
                assumptions_description="Data in both groups are normally distributed.",
                long_description=f"Group 1 label: '{categories[0]}'. "
                f"Group 2 label: '{categories[1]}'.",
            )

        elif test_type == "yuen":
            ttest_result = stats.ttest_ind(
                group_1, group_2, equal_var=False, trim=0.1, alternative="two-sided"
            )
            return StatisticalTestResult(
                description="Yuen's (20% trimmed mean) t-test",
                statistic=ttest_result.statistic,
                pval=ttest_result.pvalue,
                descriptive_statistic=float(group_1.mean() - group_2.mean()),
                degfree=ttest_result.df,
                statistic_description="t-statistic",
                descriptive_statistic_description="mu_1 - mu_2",
                null_hypothesis_description="mu_1 = mu_2",
                alternative_hypothesis_description="mu_1 != mu_2",
                long_description=f"Group 1 label: '{categories[0]}'. "
                f"Group 2 label: '{categories[1]}'. "
                "Yuen's test is a robust alternative to Welch's "
                "t-test when the assumption of homogeneity of variance "
                "is violated. "
                "10% of the data is trimmed from each tail.",
            )

        elif test_type == "mann-whitney":
            u_stat, p_val = stats.mannwhitneyu(
                group_1, group_2, alternative="two-sided"
            )
            return StatisticalTestResult(
                description="Mann-Whitney U test",
                statistic=u_stat,
                pval=p_val,
                descriptive_statistic=None,
                degfree=None,
                statistic_description="U-statistic",
                descriptive_statistic_description=None,
                null_hypothesis_description="mu_1 = mu_2",
                alternative_hypothesis_description="mu_1 != mu_2",
                assumptions_description="Var(Group 1) = Var(Group 2).",
                long_description=f"Group 1 label: '{categories[0]}'. "
                f"Group 2 label: '{categories[1]}'. "
                "Mann-Whitney U test is a non-parametric test for "
                "testing the null hypothesis that the distributions "
                "of two independent samples are equal.",
            )

    # --------------------------------------------------------------------------
    # GETTERS
    # --------------------------------------------------------------------------
    def numeric_vars(self) -> list[str]:
        """Returns a list of the names of all numeric variables.

        Returns
        -------
        list[str]
        """
        return self._numeric_vars

    def categorical_vars(self) -> list[str]:
        """Returns a list of the names of all categorical variables.

        Returns
        -------
        list[str]
        """
        return self._categorical_vars

    def categorical_stats(self) -> pd.DataFrame | None:
        """Returns a DataFrame containing summary statistics for all
        categorical variables.

        Returns None if there are no categorical variables.

        Returns
        -------
        pd.DataFrame | None
        """
        return self._categorical_summary_statistics

    def numeric_stats(self) -> pd.DataFrame | None:
        """Returns a DataFrame containing summary statistics for all
        numeric variables.

        Returns None if there are no numeric variables.

        Returns
        -------
        pd.DataFrame | None
        """
        return self._numeric_summary_statistics

    def specific(self, var: str) -> CategoricalEDA | NumericEDA:
        """Returns the CategoricalEDA or NumericEDA object associated with
        the input variable.

        Parameters
        ----------
        var : str

        Returns
        -------
        CategoricalEDA | NumericEDA
        """
        if var in self._categorical_vars:
            return self._categorical_eda_dict[var]
        elif var in self._numeric_vars:
            return self._numeric_eda_dict[var]
        else:
            raise ValueError(
                f"Invalid input: {var}. " + "Must be a known variable in the input df."
            )

    def _agentic_describe_json_str(self) -> str:
        """Returns a jsonified string representation of the dataset.

        Returns
        -------
        str
        """
        output = {}
        output[
            "categorical variable summary statistics"
        ] = self._categorical_summary_statistics.to_dict()
        output[
            "numeric variable summary statistics"
        ] = self._numeric_summary_statistics.to_dict()
        output["number of numeric variables"] = len(self._numeric_vars)
        output["number of categorical variables"] = len(self._categorical_vars)
        output["number of examples/rows"] = len(self._df)
        return json.dumps(output)

    def __getitem__(self, index: str) -> CategoricalEDA | NumericEDA:
        """Indexes into EDAReport.

        Parameters
        ----------
        index : str

        Returns
        -------
        CategoricalEDA | NumericEDA
        """
        if isinstance(index, str):
            if index in self._categorical_vars:
                return self._categorical_eda_dict[index]
            elif index in self._numeric_vars:
                return self._numeric_eda_dict[index]
            else:
                raise ValueError(
                    f"Invalid input: {index}. Index must be a "
                    + "variable name in the input df."
                )
        else:
            raise ValueError(f"Invalid input: {index}. Index must be a string.")
