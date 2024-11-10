import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Literal
from sklearn.preprocessing import minmax_scale, scale
from sklearn.decomposition import PCA
from textwrap import fill
from ..stattests import StatisticalTestReport
from ..display.print_utils import print_wrapped, format_value
from ..display.print_options import print_options
from ..display.plot_options import plot_options
from ..utils import ensure_arg_list_uniqueness


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Safely compute Pearson correlation with missing value handling."""
    # Get mask for pairwise complete observations
    mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(mask) < 2:  # Need at least 2 complete pairs
        return np.nan, np.nan
    try:
        return stats.pearsonr(x[mask], y[mask])
    except Exception:  # Catch any numerical computation errors
        return np.nan, np.nan


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

        ax.bar(
            value_freqs.index,
            value_freqs.values,
            color=plot_options._bar_color,
            edgecolor=plot_options._bar_edgecolor,
            alpha=plot_options._bar_alpha,
        )
        ax.set_title(
            f"Distrubution of {self.variable_name}",
        )
        ax.set_xlabel(
            "Categories",
        )

        if density:
            ax.set_ylabel(
                "Density",
            )
        else:
            ax.set_ylabel(
                "Frequency",
            )
        ax.ticklabel_format(style="sci", axis="y", scilimits=plot_options._scilimits)

        ax.title.set_fontsize(plot_options._title_font_size)
        ax.xaxis.label.set_fontsize(plot_options._axis_title_font_size)
        ax.yaxis.label.set_fontsize(plot_options._axis_title_font_size)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=plot_options._axis_major_ticklabel_font_size,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            labelsize=plot_options._axis_minor_ticklabel_font_size,
        )

        if fig is not None:
            fig.tight_layout()
            plt.close()
        return fig

    def counts(self, normalize: bool = False) -> pd.Series:
        """Returns the counts of each category in the variable.

        Parameters
        ----------
        normalize : bool
            Default: False. If True, returns the relative frequencies
            of the categories

        Returns
        -------
        pd.Series
        """
        return self._var_series.value_counts(normalize=normalize)


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
        """Returns a histogram Figure.

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
            color=plot_options._bar_color,
            edgecolor=plot_options._bar_edgecolor,
            stat=stat,
            ax=ax,
            kde=True,
            alpha=plot_options._bar_alpha,
        )
        ax.set_title(f"Distribution of {self.variable_name}")
        ax.set_xlabel("Values")
        if density:
            ax.set_ylabel("Density")
        else:
            ax.set_ylabel("Frequency")
        ax.ticklabel_format(style="sci", axis="both", scilimits=plot_options._scilimits)

        ax.title.set_fontsize(plot_options._title_font_size)
        ax.xaxis.label.set_fontsize(plot_options._axis_title_font_size)
        ax.yaxis.label.set_fontsize(plot_options._axis_title_font_size)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=plot_options._axis_major_ticklabel_font_size,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            labelsize=plot_options._axis_minor_ticklabel_font_size,
        )

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
            ).T
            self._categorical_summary_statistics["n"] = (
                self._categorical_summary_statistics["n"].astype(int)
            )
            self._categorical_summary_statistics["n_missing"] = (
                self._categorical_summary_statistics["n_missing"].astype(int)
            )
            self._categorical_summary_statistics.index.name = "Variable"

        if len(self._numeric_vars) > 0:
            self._numeric_summary_statistics = pd.concat(
                [eda.summary_statistics for eda in self._numeric_eda_dict.values()],
                axis=1,
            ).T
            self._numeric_summary_statistics["n"] = self._numeric_summary_statistics[
                "n"
            ].astype(int)
            self._numeric_summary_statistics["n_missing"] = (
                self._numeric_summary_statistics["n_missing"].astype(int)
            )
            self._numeric_summary_statistics.index.name = "Variable"

    # --------------------------------------------------------------------------
    # TABLE GENERATION
    # --------------------------------------------------------------------------
    @ensure_arg_list_uniqueness()
    def tabulate_correlation_comparison(
        self, numeric_vars: list[str], target: str, bonferroni_correction: bool = False
    ) -> pd.DataFrame:
        """Generates a table of the Pearson correlation coefficients between the
        numeric variables and a target variable.

        Parameters
        ----------
        numeric_vars : list[str]
            List of numeric variables.
        target : str
            The numeric variable to correlate the `numeric_vars` with.
        bonferroni_correction : bool, default=False
            If True, applies the Bonferroni correction to the p-values
            (multiplies them by the number of tests).
        dropna : bool, default=True
            If True, drops rows with NaN values when computing correlations.
            If False, raises an error if NaN values are present.

        Returns
        -------
        pd.DataFrame
            DataFrame with index as the numeric variables.
            Columns include the Pearson correlation coefficient, p-value, and
            number of units considered (if dropna was True).
        """
        # Input validation
        invalid_vars = set(numeric_vars) - set(self._numeric_vars)
        if invalid_vars:
            raise ValueError(
                f"Invalid input(s): {', '.join(invalid_vars)}. "
                "Must be known numeric variables."
            )
        if target not in self._numeric_vars:
            raise ValueError(
                f"Invalid input: {target}. Must be a known numeric variable."
            )
        if bonferroni_correction:
            p_val_name = f"p-value (Bonferroni corrected, n_tests={len(numeric_vars)})"
        else:
            p_val_name = "p-value"

        corr_column_name = f"Corr. w {target}"

        corr_table = pd.DataFrame(columns=[corr_column_name, p_val_name, "n"])

        has_missing = self._df[numeric_vars + [target]].isnull().any(axis=1).any()

        for var in numeric_vars:
            var_data = self._df[var]
            target_data = self._df[target]

            corr, p = safe_pearsonr(var_data, target_data)
            if bonferroni_correction:
                p *= len(numeric_vars)
            corr_table.loc[var] = [corr, p, len(var_data)]

        n_decimals = getattr(print_options, "_n_decimals", 4)

        corr_table[corr_column_name] = corr_table[corr_column_name].apply(
            lambda x: format_value(x, n_decimals)
        )
        corr_table[p_val_name] = corr_table[p_val_name].apply(
            lambda x: format_value(x, n_decimals)
        )

        if not has_missing:
            corr_table = corr_table.drop("n", axis=1)
        else:
            corr_table["n"] = corr_table["n"].astype(int)

        return corr_table

    @ensure_arg_list_uniqueness()
    def tabulate_correlation_matrix(
        self,
        numeric_vars: list[str],
        p_values: bool = False,
    ) -> pd.DataFrame:
        """Generates a table of the Pearson correlation coefficients
        between numeric variables.

        The function computes correlations efficiently by leveraging numpy operations
        and avoiding redundant calculations. For symmetric pairs (i,j) and (j,i),
        it only computes one and mirrors the result. Handles missing values by using
        pairwise complete observations.

        Parameters
        ----------
        numeric_vars : list[str]
            List of numeric variables to compute correlations for.

        p_values : bool, default=False
            If True, includes p-values in the output in format: "corr (p-val)"

        Returns
        -------
        pd.DataFrame
            DataFrame with index and columns as the numeric variables.
            Values are either correlation coefficients or "correlation (p-value)" if
            p_values=True. Missing values are represented as "NA".

        Raises
        ------
        ValueError
            If any variable in numeric_vars is not a known numeric variable.
        """

        def format_corr_pval(corr: float, pval: float, n_decimals: int = 4) -> str:
            """Format correlation and p-value pair with consistent decimal places."""
            if pd.isna(corr) or pd.isna(pval):
                return "NA"
            corr_str = format_value(corr, n_decimals)
            pval_str = format_value(pval, n_decimals)
            return f"{corr_str} ({pval_str})"

        # Validate input variables
        invalid_vars = set(numeric_vars) - set(self._numeric_vars)
        if invalid_vars:
            raise ValueError(
                f"Invalid input(s): {', '.join(invalid_vars)}. "
                "Must be known numeric variables."
            )

        # Extract data as numpy array for faster computation
        data = self._df[numeric_vars].values
        n_vars = len(numeric_vars)
        n_decimals = getattr(print_options, "_n_decimals", 4)

        # Initialize matrices for correlations and p-values
        corr_matrix = np.ones((n_vars, n_vars))
        p_matrix = np.ones((n_vars, n_vars)) if p_values else None

        # Compute correlations and p-values for upper triangle
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                corr, p = safe_pearsonr(data[:, i], data[:, j])
                corr_matrix[i, j] = corr_matrix[j, i] = corr
                if p_values:
                    p_matrix[i, j] = p_matrix[j, i] = p

        # Convert to DataFrame and format values
        if not p_values:
            # Format correlation values only
            def format_value_with_na(x):
                return "NA" if pd.isna(x) else format_value(x, n_decimals)

            formatted_matrix = np.vectorize(format_value_with_na)(corr_matrix)
            result = pd.DataFrame(
                formatted_matrix, index=numeric_vars, columns=numeric_vars
            )
        else:
            # Format correlations with p-values
            formatted_matrix = np.empty((n_vars, n_vars), dtype=object)
            for i in range(n_vars):
                for j in range(n_vars):
                    if i == j:
                        formatted_matrix[i, j] = (
                            format_value(1.0, n_decimals)
                            + f" ({format_value(1.0, n_decimals)})"
                        )
                    else:
                        formatted_matrix[i, j] = format_corr_pval(
                            corr_matrix[i, j], p_matrix[i, j], n_decimals
                        )

            result = pd.DataFrame(
                formatted_matrix, index=numeric_vars, columns=numeric_vars
            )

        return result

    def tabulate_tableone(self, vars: list[str], stratify_by: str) -> pd.DataFrame:
        """
        Generates a tableone for the given variables stratified by the given variable.

        Parameters
        ----------
        vars : list[str]
            List of variables to include in the tableone.

        stratify_by : str
            The variable to stratify by.

        Returns
        -------
        pd.DataFrame
        """

    # --------------------------------------------------------------------------
    # PLOTTING
    # --------------------------------------------------------------------------
    @ensure_arg_list_uniqueness()
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
        font_adjustment = 1

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
                palette=plot_options._color_palette[
                    : len(self._df[stratify_by].unique())
                ],
            )
            right_adjust = 0.85

        grid.map_diag(
            sns.histplot,
            color=plot_options._bar_color,
            edgecolor=plot_options._bar_edgecolor,
            kde=True,
            alpha=plot_options._bar_alpha,
        )

        if stratify_by is None:
            grid.map_upper(
                lambda x, y, **kwargs: plt.text(
                    0.5,
                    0.5,
                    f"ρ = {round(stats.pearsonr(x, y)[0], print_options._n_decimals)}\n"
                    + f"p = {round(stats.pearsonr(x, y)[1], print_options._n_decimals)}",
                    ha="center",
                    va="center",
                    transform=plt.gca().transAxes,
                    fontsize=plot_options._axis_title_font_size - font_adjustment,
                )
            )
        else:
            grid.map_upper(
                sns.scatterplot, s=plot_options._dot_size, color=plot_options._dot_color
            )

        grid.map_lower(
            sns.scatterplot, s=plot_options._dot_size, color=plot_options._dot_color
        )

        grid.add_legend()

        for ax in grid.axes.flat:
            ax.ticklabel_format(
                style="sci", axis="both", scilimits=plot_options._scilimits
            )

            ax.xaxis.offsetText.set_fontsize(
                plot_options._axis_major_ticklabel_font_size - font_adjustment
            )
            ax.yaxis.offsetText.set_fontsize(
                plot_options._axis_major_ticklabel_font_size - font_adjustment
            )

            ax.xaxis.label.set_fontsize(
                plot_options._axis_title_font_size - font_adjustment
            )
            ax.yaxis.label.set_fontsize(
                plot_options._axis_title_font_size - font_adjustment
            )
            ax.tick_params(
                axis="both",
                which="major",
                labelsize=plot_options._axis_major_ticklabel_font_size
                - font_adjustment,
            )
            ax.tick_params(
                axis="both",
                which="minor",
                labelsize=plot_options._axis_minor_ticklabel_font_size
                - font_adjustment,
            )

        fig = grid.figure
        fig.set_size_inches(*figsize)
        fig.subplots_adjust(wspace=0.2, hspace=0.2, right=right_adjust)

        if stratify_by is not None:
            legend = fig.legends[0]
            legend.set_title(
                legend.get_title().get_text(),
                prop={"size": plot_options._axis_title_font_size},
            )
            for text in legend.get_texts():
                text.set_text(fill(text.get_text(), 10))
                text.set_fontsize(plot_options._axis_title_font_size - font_adjustment)
        fig.subplots_adjust(left=0.1, bottom=0.1)
        plt.close(fig)
        return fig

    @ensure_arg_list_uniqueness()
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
                        alpha=plot_options._bar_alpha,
                        stat="density",
                        ax=ax,
                        color=plot_options._color_palette[i],
                        edgecolor=plot_options._bar_edgecolor,
                    )
                elif strategy == "stacked_hist_kde_frequency":
                    sns.histplot(
                        subset[numeric_var],
                        bins="auto",
                        kde=True,
                        label=str(category),
                        alpha=plot_options._bar_alpha,
                        stat="frequency",
                        ax=ax,
                        color=plot_options._color_palette[i],
                        edgecolor=plot_options._bar_edgecolor,
                    )
                elif strategy == "stacked_kde_density":
                    sns.kdeplot(subset[numeric_var], label=str(category), ax=ax)

            legend = ax.legend()
            legend.set_title(stratify_by)

            ax.ticklabel_format(
                style="sci", axis="x", scilimits=plot_options._scilimits
            )

        elif strategy == "violin_swarm":
            sns.violinplot(
                data=local_df,
                x=stratify_by,
                y=numeric_var,
                ax=ax,
                inner=None,
                color=plot_options._bar_color,
                edgecolor=plot_options._bar_edgecolor,
                alpha=plot_options._bar_alpha,
            )
            sns.swarmplot(
                data=local_df,
                x=stratify_by,
                y=numeric_var,
                color=plot_options._dot_color,
                size=plot_options._dot_size,
                ax=ax,
            )

        elif strategy == "violin":
            sns.violinplot(
                data=local_df,
                x=stratify_by,
                y=numeric_var,
                ax=ax,
                color=plot_options._bar_color,
                edgecolor=plot_options._bar_color,
                inner="box",
                alpha=plot_options._bar_alpha,
            )

        elif strategy == "violin_strip":
            sns.violinplot(
                data=local_df,
                x=stratify_by,
                y=numeric_var,
                ax=ax,
                inner=None,
                color=plot_options._bar_color,
                edgecolor=plot_options._bar_edgecolor,
                alpha=plot_options._bar_alpha,
            )
            sns.stripplot(
                data=local_df,
                x=stratify_by,
                y=numeric_var,
                color=plot_options._dot_color,
                size=plot_options._dot_size,
                ax=ax,
            )

        elif strategy == "box_swarm":
            sns.boxplot(
                data=local_df,
                x=stratify_by,
                y=numeric_var,
                ax=ax,
                color=plot_options._bar_color,
                edgecolor=plot_options._bar_edgecolor,
                fill=False,
                linewidth=0.5,
            )
            sns.swarmplot(
                data=local_df,
                x=stratify_by,
                y=numeric_var,
                color=plot_options._dot_color,
                size=plot_options._dot_size,
                ax=ax,
            )

        elif strategy == "box_strip":
            sns.boxplot(
                data=local_df,
                x=stratify_by,
                y=numeric_var,
                ax=ax,
                color=plot_options._bar_color,
                fill=False,
                linewidth=0.5,
            )
            sns.stripplot(
                data=local_df,
                x=stratify_by,
                y=numeric_var,
                color=plot_options._dot_color,
                size=plot_options._dot_size,
                ax=ax,
            )

        elif strategy == "box":
            sns.boxplot(
                data=local_df,
                x=stratify_by,
                y=numeric_var,
                ax=ax,
                color=plot_options._bar_color,
                fill=False,
                linewidth=0.5,
            )

        else:
            raise ValueError(f"Invalid input: {strategy}.")

        ax.ticklabel_format(style="sci", axis="y", scilimits=plot_options._scilimits)
        ax.set_title(
            f"Distribution of {numeric_var}",
        )

        ax.title.set_fontsize(plot_options._title_font_size)
        ax.xaxis.label.set_fontsize(plot_options._axis_title_font_size)
        ax.yaxis.label.set_fontsize(plot_options._axis_title_font_size)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=plot_options._axis_major_ticklabel_font_size,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            labelsize=plot_options._axis_minor_ticklabel_font_size,
        )

        if fig is not None:
            fig.tight_layout()
            plt.close()
        return fig

    @ensure_arg_list_uniqueness()
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

    @ensure_arg_list_uniqueness()
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
            for color, category in zip(
                plot_options._color_palette, np.unique(categories)
            ):
                mask = categories == category
                if three_components:
                    ax.scatter(
                        components[mask, 0],
                        components[mask, 1],
                        components[mask, 2],
                        label=category,
                        s=plot_options._dot_size,
                        color=color,
                    )
                    ax.set_xlabel("Principle Component 1")
                    ax.set_ylabel("Principle Component 2")
                    ax.set_zlabel("Principle Component 3")
                else:
                    ax.scatter(
                        components[mask, 0],
                        components[mask, 1],
                        label=category,
                        s=plot_options._dot_size,
                        color=color,
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
            for color, category in zip(
                plot_options._color_palette, np.unique(categories)
            ):
                mask = categories == category
                if three_components:
                    ax.scatter(
                        components[mask, 0],
                        components[mask, 1],
                        components[mask, 2],
                        label=category,
                        s=plot_options._dot_size,
                        color=color,
                    )
                    ax.set_xlabel("Principle Component 1")
                    ax.set_ylabel("Principle Component 2")
                    ax.set_zlabel("Principle Component 3")
                else:
                    ax.scatter(
                        components[mask, 0],
                        components[mask, 1],
                        label=category,
                        s=plot_options._dot_size,
                        color=color,
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
                    components[:, 0],
                    components[:, 1],
                    components[:, 2],
                    color=plot_options._dot_color,
                    s=plot_options._dot_size,
                )
                ax.set_xlabel("Principle Component 1")
                ax.set_ylabel("Principle Component 2")
                ax.set_zlabel("Principle Component 3")
            else:
                ax.scatter(
                    components[:, 0],
                    components[:, 1],
                    color=plot_options._dot_color,
                    s=plot_options._dot_size,
                )
                ax.set_xlabel("Principle Component 1")
                ax.set_ylabel("Principle Component 2")

        title_str = ", ".join(numeric_vars)
        ax.set_title(f"PCA({title_str})", wrap=True)
        ax.ticklabel_format(style="sci", axis="both", scilimits=(-3, 3))

        ax.title.set_fontsize(plot_options._title_font_size)
        ax.xaxis.label.set_fontsize(plot_options._axis_title_font_size)
        ax.yaxis.label.set_fontsize(plot_options._axis_title_font_size)
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=plot_options._axis_major_ticklabel_font_size,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            labelsize=plot_options._axis_minor_ticklabel_font_size,
        )

        legend = ax.legend_
        legend.set_title(
            legend.get_title().get_text(),
            prop={"size": plot_options._axis_title_font_size},
        )
        for text in legend.get_texts():
            text.set_fontsize(plot_options._axis_title_font_size)

        if fig is not None:
            if not three_components:
                fig.tight_layout()
            else:
                fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            plt.close()
        return fig

    @ensure_arg_list_uniqueness()
    def plot_correlation_heatmap(
        self,
        numeric_vars: list[str] | None = None,
        p_values: bool = False,
        cmap: str | plt.Colormap = "coolwarm",
        figsize: tuple[float, float] = (5, 5),
        ax: plt.Axes | None = None,
    ) -> plt.Figure:
        """Plots a heatmap of the correlation matrix of the numeric variables.

        Parameters
        ----------
        numeric_vars : list[str] | None
            List of numeric variables to include in the heatmap.
            If None, all numeric variables are considered.

        p_values : bool
            If True, displays correlation coefficients with their
            corresponding p-values in parentheses.

        cmap : str | plt.Colormap
            The colormap to use for the heatmap visualization.

        figsize : tuple[float, float]
            The size of the figure (width, height) in inches.
            Only used if ax is None.

        ax : plt.Axes | None
            If provided, the plot is drawn on this Axes instance.

        Returns
        -------
        plt.Figure
            The figure containing the correlation heatmap.
        """

        def format_corr_pval(corr: float, pval: float, n_decimals: int = 4) -> str:
            """Format correlation and p-value pair."""
            if pd.isna(corr) or pd.isna(pval):
                return "NA"
            corr_str = format_value(corr, n_decimals)
            pval_str = format_value(pval, n_decimals)
            return f"{corr_str}\n(p={pval_str})"

        if numeric_vars is None:
            numeric_vars = self._numeric_vars
        else:
            invalid_vars = set(numeric_vars) - set(self._numeric_vars)
            if invalid_vars:
                raise ValueError(
                    f"Invalid input(s): {', '.join(invalid_vars)}. "
                    "Must be known numeric variables."
                )

        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        n_decimals = getattr(print_options, "_n_decimals", 4)

        if not p_values:
            # Use pandas corr() which handles missing values with pairwise deletion
            corr = self._df[numeric_vars].corr()

            # Format the correlation values, handling NaN
            def format_value_with_na(x):
                return "NA" if pd.isna(x) else format_value(x, n_decimals)

            corr_formatted = corr.applymap(format_value_with_na)

            # Create a mask for missing values to show them differently
            mask = corr.isna()

            sns.heatmap(
                corr,
                annot=corr_formatted,
                annot_kws={
                    "size": plot_options._axis_title_font_size,
                    "ha": "center",
                    "va": "center",
                },
                cmap=cmap,
                ax=ax,
                fmt="",
                cbar=True,
                vmin=-1,  # Force scale from -1 to 1
                vmax=1,
                center=0,  # Center colormap at 0
                mask=mask,  # Mask NA values
            )
        else:
            # initialize matrices for correlations and p-values
            n_vars = len(numeric_vars)
            corr_matrix = np.ones((n_vars, n_vars))
            p_matrix = np.ones((n_vars, n_vars))
            annot_matrix = np.empty((n_vars, n_vars), dtype=object)

            # compute correlations and p-values for all pairs
            for i in range(n_vars):
                for j in range(n_vars):
                    if i == j:
                        # handle diagonal
                        corr_matrix[i, j] = 1.0
                        p_matrix[i, j] = 1.0
                        annot_matrix[i, j] = (
                            format_value(1.0, n_decimals) + "\n(p=1.0000)"
                        )
                    else:
                        corr, p = safe_pearsonr(
                            self._df[numeric_vars[i]], self._df[numeric_vars[j]]
                        )
                        corr_matrix[i, j] = corr
                        p_matrix[i, j] = p
                        annot_matrix[i, j] = format_corr_pval(corr, p, n_decimals)

            # Create mask for NA values
            mask = np.isnan(corr_matrix)

            sns.heatmap(
                corr_matrix,
                annot=annot_matrix,
                annot_kws={
                    "size": plot_options._axis_title_font_size,
                    "ha": "center",
                    "va": "center",
                },
                fmt="",
                cmap=cmap,
                ax=ax,
                cbar=True,
                vmin=-1,
                vmax=1,
                center=0,
                mask=mask,  # Mask NA values
            )

        # customize appearance
        title = "Correlation Heatmap"
        ax.set_title(title, fontsize=plot_options._title_font_size)

        ax.set_xticklabels(
            numeric_vars,
            rotation=45,
            ha="right",
            fontsize=plot_options._axis_major_ticklabel_font_size,
        )
        ax.set_yticklabels(
            numeric_vars,
            rotation=0,
            fontsize=plot_options._axis_major_ticklabel_font_size,
        )

        if fig is not None:
            fig.tight_layout()
            plt.close()

        return fig

    # --------------------------------------------------------------------------
    # STATISTICAL TESTING
    # --------------------------------------------------------------------------

    def test_equal_means(
        self, numeric_var: str, stratify_by: str
    ) -> StatisticalTestReport:
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
    ) -> StatisticalTestReport:
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
            return StatisticalTestReport(
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

            return StatisticalTestReport(
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
    ) -> StatisticalTestReport:
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

        group_1_str = f"{categories[0]}::{stratify_by}"
        group_2_str = f"{categories[1]}::{stratify_by}"

        group_1_full_str = f"{numeric_var}_{group_1_str}"
        group_2_full_str = f"{numeric_var}_{group_2_str}"

        mu_1_str = f"Mean({group_1_full_str})"
        mu_2_str = f"Mean({group_2_full_str})"

        if test_type == "student":
            ttest_result = stats.ttest_ind(
                group_1, group_2, equal_var=True, alternative="two-sided"
            )
            return StatisticalTestReport(
                description="Student's t-test",
                statistic=ttest_result.statistic,
                pval=ttest_result.pvalue,
                descriptive_statistic=float(group_1.mean() - group_2.mean()),
                degfree=ttest_result.df,
                statistic_description="t-statistic",
                descriptive_statistic_description=f"{mu_1_str} - {mu_2_str}",
                null_hypothesis_description=f"{mu_1_str} = {mu_2_str}",
                alternative_hypothesis_description=f"{mu_1_str} != {mu_2_str}",
                assumptions_description=[
                    f"Var({group_1_full_str}) = Var({group_2_full_str}).",
                    f"Values for {numeric_var} in groups {group_1_str} and "
                    f"{group_2_str} are normally distributed.",
                ],
            )

        elif test_type == "welch":
            ttest_result = stats.ttest_ind(
                group_1, group_2, equal_var=False, alternative="two-sided"
            )
            return StatisticalTestReport(
                description="Welch's t-test",
                statistic=ttest_result.statistic,
                pval=ttest_result.pvalue,
                descriptive_statistic=float(group_1.mean() - group_2.mean()),
                degfree=ttest_result.df,
                statistic_description="t-statistic",
                descriptive_statistic_description=f"{mu_1_str} - {mu_2_str}",
                null_hypothesis_description=f"{mu_1_str} = {mu_2_str}",
                alternative_hypothesis_description=f"{mu_1_str} != {mu_2_str}",
                assumptions_description=f"Values for {numeric_var} in groups {group_1_str} and "
                f"{group_2_str} are normally distributed.",
            )

        elif test_type == "yuen":
            ttest_result = stats.ttest_ind(
                group_1, group_2, equal_var=False, trim=0.1, alternative="two-sided"
            )
            return StatisticalTestReport(
                description="Yuen's (20% trimmed mean) t-test",
                statistic=ttest_result.statistic,
                pval=ttest_result.pvalue,
                descriptive_statistic=float(group_1.mean() - group_2.mean()),
                degfree=ttest_result.df,
                statistic_description="t-statistic",
                descriptive_statistic_description=f"{mu_1_str} - {mu_2_str}",
                null_hypothesis_description=f"{mu_1_str} = {mu_2_str}",
                alternative_hypothesis_description=f"{mu_1_str} != {mu_2_str}",
                long_description="Yuen's test is a robust alternative to Welch's "
                "t-test when the assumption of homogeneity of variance "
                "is violated. "
                "For both groups, 10% of the most extreme observations are trimmed "
                "from each tail.",
            )

        elif test_type == "mann-whitney":
            u_stat, p_val = stats.mannwhitneyu(
                group_1, group_2, alternative="two-sided"
            )
            return StatisticalTestReport(
                description="Mann-Whitney U test",
                statistic=u_stat,
                pval=p_val,
                descriptive_statistic=float(group_1.mean() - group_2.mean()),
                degfree=None,
                statistic_description="U-statistic",
                descriptive_statistic_description=f"{mu_1_str} - {mu_2_str}",
                null_hypothesis_description=f"{mu_1_str} = {mu_2_str}",
                alternative_hypothesis_description=f"{mu_1_str} != {mu_2_str}",
                assumptions_description=f"Var({group_1_full_str}) = Var({group_2_full_str}).",
                long_description="Mann-Whitney U test is a non-parametric test for "
                "testing the null hypothesis that the distributions "
                "of two independent samples are equal.",
            )

    # --------------------------------------------------------------------------
    # GETTERS (as methods, not properties)
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
        return self._categorical_summary_statistics.round(print_options._n_decimals)

    def numeric_stats(self) -> pd.DataFrame | None:
        """Returns a DataFrame containing summary statistics for all
        numeric variables.

        Returns None if there are no numeric variables.

        Returns
        -------
        pd.DataFrame | None
        """
        return self._numeric_summary_statistics.round(print_options._n_decimals)

    def value_counts(
        self, categorical_var: str, normalize: bool = False
    ) -> pd.DataFrame:
        """Returns the value counts for a given categorical variable as a
        DataFrame, with first column as the unique values and the second
        column as the counts.

        Parameters
        ----------
        categorical_var : str
            Categorical variable name.

        normalize : bool
            Default: False. If True, returns the value counts as proportions.

        Returns
        -------
        pd.DataFrame
        """
        if categorical_var not in self._categorical_vars:
            raise ValueError(
                f"Invalid input: {categorical_var}. "
                + "Must be a known categorical variable."
            )
        return pd.DataFrame(
            self._categorical_eda_dict[categorical_var].counts(normalize)
        ).reset_index()

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
