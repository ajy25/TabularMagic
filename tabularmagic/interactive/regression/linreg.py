import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import seaborn as sns
from scipy import stats
from typing import Iterable, Literal
from ...metrics.regression_scoring import RegressionScorer
from ...preprocessing.datapreprocessor import BaseSingleVarScaler
from ..visualization import plot_pred_vs_true, decrease_font_sizes_axs
from ...linear.regression.linear_regression import OrdinaryLeastSquares
from adjustText import adjust_text


def reverse_argsort(indices):
    n = len(indices)
    reverse_indices = [0] * n
    for i, idx in enumerate(indices):
        reverse_indices[idx] = i
    return reverse_indices

train_only_message = 'This function is only available for training data.'


class LinearRegressionReport():
    """LinearRegressionReport: generates regression-relevant diagnostic 
    plots and tables for a single linear regression model. 
    """
    
    def __init__(self, model: OrdinaryLeastSquares, X_eval: pd.DataFrame, 
                 y_eval: pd.DataFrame, y_scaler: BaseSingleVarScaler = None):
        """
        Initializes a RegressionReport object. 

        Parameters
        ----------
        - model : BaseRegression.
            The model must already be trained.
        - X_eval : pd.DataFrame.
        - y_eval : pd.Series.
        - y_scaler: BaseSingleVarScaler.
            Default: None. If exists, calls inverse transform on the outputs 
            and on y_eval before computing statistics.
        """
        self.model = model
        self._X_eval_df = X_eval
        self._y_eval_df = y_eval
        self._y_pred = model.predict(self._X_eval_df)
        self._is_train = False
        if len(self._y_pred) == len(model.estimator.fittedvalues):
            if np.allclose(self._y_pred, model.estimator.fittedvalues):
                self._is_train = True
        self._y_true = self._y_eval_df.to_numpy()
        if y_scaler is not None:
            self._y_pred = y_scaler.inverse_transform(self._y_pred)
            self._y_true = y_scaler.inverse_transform(self._y_true)
        self._scorer = RegressionScorer(y_pred=self._y_pred, y_true=self._y_true, 
            n_regressors=model._n_regressors, model_id_str=str(model))
        self._residuals = self._y_true - self._y_pred
        self._stdresiduals = self._residuals / np.std(self._residuals)
        self._outlier_threshold = 2
        self._compute_outliers()

        
    def statsmodels_summary(self):
        """
        Returns the summary of the statsmodels RegressionResultsWrapper for 
        OLS.
        """
        try:
            return self.model.estimator.summary()
        except:
            raise RuntimeError('Error occured in statsmodels_summary call.')


    def plot_pred_vs_true(self, show_outliers: bool = True,
                          figsize: Iterable = (5, 5), 
                          ax: axes.Axes = None):
        """Returns a figure that is a scatter plot of the true and predicted y 
        values. 

        Parameters 
        ----------
        - figsize : Iterable. 
        - ax : axes.Axes.

        Returns
        -------
        - plt.Figure
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        plot_pred_vs_true(self._y_pred, self._y_true, figsize, ax)
        if show_outliers:
            ax.scatter(self._y_pred[self._outliers_residual_mask], 
                    self._y_true[self._outliers_residual_mask], s=2, 
                    color='red')
            annotations = []
            for i, label in enumerate(self._outliers_df_idx):
                annotations.append(
                    ax.annotate(label, 
                    (self._y_pred[self._outliers_residual_mask][i], 
                    self._y_true[self._outliers_residual_mask][i]), 
                    color='red',
                    fontsize=6))
            adjust_text(annotations, ax=ax)
                
                
        if fig is not None:
            fig.tight_layout()
            plt.close()
        return fig
    

    def plot_residuals_vs_fitted(self, standardized: bool = False, 
                                 show_outliers: bool = True,
                                 figsize: Iterable = (5, 5), 
                                 ax: axes.Axes = None):
        """Returns a figure that is a residuals vs fitted (y_pred) plot. 

        Parameters
        ----------
        - standardized : bool. If True, standardizes the residuals. 
        - show_outliers : bool. If True, plots the outliers in red.
        - figsize : Iterable.
        - ax : axes.Axes

        Returns
        -------
        - plt.Figure
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        residuals = self._residuals
        if standardized:
            residuals = self._stdresiduals

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        if show_outliers and len(self._outliers_residual_mask) > 0:
            ax.scatter(self._y_pred[~self._outliers_residual_mask], 
                    residuals[~self._outliers_residual_mask], s=2, 
                    color='black')
            ax.scatter(self._y_pred[self._outliers_residual_mask], 
                    residuals[self._outliers_residual_mask], s=2, 
                    color='red')
            annotations = []
            for i, label in enumerate(self._outliers_df_idx):
                annotations.append(
                    ax.annotate(label, 
                    (self._y_pred[self._outliers_residual_mask][i], 
                        residuals[self._outliers_residual_mask][i]), 
                    color='red', fontsize=6))
            adjust_text(annotations, ax=ax)
        else:
            ax.scatter(self._y_pred, residuals, s=2, color='black')

        ax.set_xlabel('Fitted')
        if standardized:
            ax.set_ylabel('Standardized Residuals')
            ax.set_title(f'Standardized Residuals vs Fitted')
        else:
            ax.set_ylabel('Residuals')
            ax.set_title(f'Residuals vs Fitted')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))

        if fig is not None:
            fig.tight_layout()
            plt.close()
        return fig


    def plot_residuals_vs_var(self, x_var: str, standardized: bool = False, 
                              show_outliers: bool = False,
                              figsize: Iterable = (5, 5), 
                              ax: axes.Axes = None):
        """Returns a figure that is a residuals vs fitted (y_pred) plot. 
        
        Parameters
        ----------
        - x_var : str.
        - standardized : bool. If True, standardizes the residuals. 
        - show_outliers : bool. If True, plots the outliers in red.
        - figsize : Iterable.
        - ax : axes.Axes

        Returns
        -------
        - plt.Figure
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        residuals = self._residuals
        if standardized:
            residuals = self._stdresiduals

        x_vals = self._X_eval_df[x_var].to_numpy()

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        if show_outliers and len(self._outliers_residual_mask) > 0:
            ax.scatter(x_vals[~self._outliers_residual_mask], 
                    residuals[~self._outliers_residual_mask], s=2, 
                    color='black')
            ax.scatter(x_vals[self._outliers_residual_mask], 
                    residuals[self._outliers_residual_mask], s=2, 
                    color='red')
            annotations = []
            for i, label in enumerate(self._outliers_df_idx):
                annotations.append(ax.annotate(label, 
                    (x_vals[self._outliers_residual_mask][i], 
                    residuals[self._outliers_residual_mask][i]), 
                    color='red', fontsize=6))
            adjust_text(annotations, ax=ax)
        else:
            ax.scatter(x_vals, residuals, s=2, color='black')

        ax.set_xlabel(x_var)
        if standardized:
            ax.set_ylabel('Standardized Residuals')
            ax.set_title(f'Standardized Residuals vs {x_var}')
        else:
            ax.set_ylabel('Residuals')
            ax.set_title(f'Residuals vs {x_var}')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))

        if fig is not None:
            fig.tight_layout()
            plt.close()
        return fig
    
    def plot_residuals_hist(self, standardized: bool = False,
                            density: bool = False, 
                            figsize: Iterable = (5, 5), 
                            ax: axes.Axes = None):
        """Returns a figure that is a histogram of the residuals.

        Parameters
        ----------
        - standardized : bool. If True, standardizes the residuals. 
        - density : bool. If True, plots density rather than frequency.
        - figsize : Iterable.
        - ax : axes.Axes

        Returns
        -------
        - plt.Figure
        """
        if density:
            stat = 'density'
        else:
            stat = 'count'
        
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        residuals = self._residuals
        if standardized:
            residuals = self._stdresiduals
        sns.histplot(residuals, bins='auto', color='black', 
                     edgecolor='none', 
                     stat=stat, ax=ax, kde=True, alpha=0.2)
        if standardized:
            ax.set_title(f'Distribution of Standardized Residuals')
            ax.set_xlabel('Standardized Residuals')
        else:
            ax.set_title(f'Distribution of Residuals')
            ax.set_xlabel('Residuals')
        if density:
            ax.set_ylabel('Density')
        else:
            ax.set_ylabel('Frequency')
        ax.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3))

        if fig is not None:
            fig.tight_layout()
            plt.close()
        return fig
    

    def plot_scale_location(self, show_outliers: bool = True, 
                            figsize: Iterable = (5, 5), 
                            ax: axes.Axes = None):
        """Returns a figure that is a plot of the 
        sqrt of the residuals versus the fitted.
        
        Parameters
        ----------
        - show_outliers : bool. If True, plots the outliers in red.
        - figsize : Iterable.
        - ax : axes.Axes. 

        Returns
        -------
        - plt.Figure
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        residuals = np.sqrt(np.abs(self._stdresiduals))

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        if show_outliers and len(self._outliers_residual_mask) > 0:
            ax.scatter(self._y_pred[~self._outliers_residual_mask], 
                    residuals[~self._outliers_residual_mask], s=2, 
                    color='black')
            ax.scatter(self._y_pred[self._outliers_residual_mask], 
                    residuals[self._outliers_residual_mask], s=2, 
                    color='red')
            annotations = []
            for i, label in enumerate(self._outliers_df_idx):
                annotations.append(ax.annotate(
                    label, 
                    (self._y_pred[self._outliers_residual_mask][i], 
                        residuals[self._outliers_residual_mask][i]), 
                    color='red',
                    fontsize=6
                ))
            adjust_text(annotations, ax=ax)
            
        else:
            ax.scatter(self._y_pred, residuals, s=2, color='black')

        ax.set_xlabel('Fitted')
        ax.set_ylabel('sqrt(Standardized Residuals)')
        ax.set_title(f'Scale-Location')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
        if fig is not None:
            fig.tight_layout()
            plt.close()
        return fig
    


    def plot_residuals_vs_leverage(self, standardized: bool = True, 
                                   show_outliers: bool = True,
                                   figsize: Iterable = (5, 5), 
                                   ax: axes.Axes = None):
        """Returns a figure that is a plot of the residuals versus leverage.
        
        Parameters
        ----------
        - standardized : bool. If True, standardizes the residuals. 
        - show_outliers : bool. If True, plots the outliers in red.
        - figsize : Iterable.
        - ax : axes.Axes. 

        Returns
        -------
        - plt.Figure
        """
        if not self._is_train:
            print(train_only_message)
            return None

        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        leverage = self.model.estimator._results.get_influence().hat_matrix_diag
        residuals = self._residuals
        if standardized:
            residuals = self._stdresiduals

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        if show_outliers and len(self._outliers_residual_mask) > 0:
            ax.scatter(leverage[~self._outliers_residual_mask], 
                    residuals[~self._outliers_residual_mask], s=2, 
                    color='black')
            ax.scatter(leverage[self._outliers_residual_mask], 
                    residuals[self._outliers_residual_mask], s=2, 
                    color='red')
            annotations = []
            for i, label in enumerate(self._outliers_df_idx):
                annotations.append(ax.annotate(
                    label, 
                    (leverage[self._outliers_residual_mask][i], 
                        residuals[self._outliers_residual_mask][i]), 
                    color='red',
                    fontsize=6
                ))
            adjust_text(annotations, ax=ax)

        else:
            ax.scatter(leverage, residuals, s=2, color='black')

        ax.set_xlabel('Leverage')
        if standardized:
            ax.set_ylabel('Standardized Residuals')
            ax.set_title(f'Standardized Residuals vs Leverage')
        else:
            ax.set_ylabel('Residuals')
            ax.set_title(f'Residuals vs Leverage')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))

        if fig is not None:
            fig.tight_layout()
            plt.close()
        return fig
    

    
    def plot_qq(self, standardized: bool = True, show_outliers: bool = False,
                figsize: Iterable = (5, 5), ax: axes.Axes = None):
        """Returns a quantile-quantile plot.
        
        Parameters 
        ----------
        - standardized : bool. If True, standardizes the residuals. 
        - show_outliers : bool. If True, plots the outliers in red.
        - figsize : Iterable.
        - ax : axes.Axes. 

        Returns
        -------
        - plt.Figure
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        if standardized:
            residuals = self._stdresiduals
        else:
            residuals = self._residuals

        tup1, tup2 = stats.probplot(residuals, dist='norm')
        theoretical_quantitles, ordered_vals = tup1
        slope, intercept, _ = tup2

        ax.set_title('Q-Q Plot')
        ax.set_xlabel('Theoretical Quantiles')

        if standardized:
            ax.set_ylabel('Standardized Residuals')
        else:
            ax.set_ylabel('Residuals')

        min_val = np.min(theoretical_quantitles)
        max_val = np.max(theoretical_quantitles)
        ax.plot([min_val, max_val], [min_val * slope + intercept, 
                                     max_val * slope + intercept], 
            color='gray', linestyle='--', linewidth=1)

        if show_outliers and len(self._outliers_residual_mask) > 0:

            residuals_sorted_idx = reverse_argsort(np.argsort(residuals))
            
            residuals_df = pd.DataFrame(residuals, columns=['residuals'])
            residuals_df['label'] = self._X_eval_df.index
            residuals_df['is_outlier'] = self._outliers_residual_mask
            residuals_df['theoretical_quantile'] =\
                  theoretical_quantitles[residuals_sorted_idx]
            residuals_df['ordered_value'] = ordered_vals[residuals_sorted_idx]
            residuals_df_outliers =\
                residuals_df[residuals_df['is_outlier'] == True]
            residuals_df_not_outliers =\
                residuals_df[residuals_df['is_outlier'] == False]
            
            ax.scatter(residuals_df_not_outliers['theoretical_quantile'], 
                residuals_df_not_outliers['ordered_value'], s=2, color='black')
            ax.scatter(residuals_df_outliers['theoretical_quantile'], 
                residuals_df_outliers['ordered_value'], s=2, color='red')
            
            annotations = []
            for _, row in residuals_df_outliers.iterrows():
                annotations.append(
                    ax.annotate(row['label'], 
                    (row['theoretical_quantile'], row['ordered_value']), 
                    color='red', 
                    fontsize=6
                ))
            adjust_text(annotations, ax=ax)
            
        else:
            ax.scatter(theoretical_quantitles, ordered_vals, 
                s=2, color='black')

        if fig is not None:
            fig.tight_layout()
            plt.close()
        return fig


    def plot_diagnostics(self, show_outliers: bool = False,
                         figsize: Iterable = (7, 7)):
        """Plots several useful linear regression diagnostic plots.

        Parameters 
        ----------
        - show_outliers : bool. If True, plots the residual outliers in red.
        - figsize : Iterable.

        Returns
        -------
        - plt.Figure
        """
        fig, axs = plt.subplots(2, 2, figsize=figsize)

        self.plot_pred_vs_true(show_outliers=show_outliers, ax=axs[0][0])
        self.plot_residuals_vs_fitted(show_outliers=show_outliers, ax=axs[0][1])

        if self._is_train:
            self.plot_residuals_vs_leverage(show_outliers=show_outliers, 
                                            ax=axs[1][0])
        else:
            self.plot_scale_location(show_outliers=show_outliers, 
                                     ax=axs[1][0])

        self.plot_qq(show_outliers=show_outliers, ax=axs[1][1])
        
        fig.subplots_adjust(hspace=0.2, wspace=0.2)

        decrease_font_sizes_axs(axs, 5, 5, 0)

        plt.close()
        return fig


    

    def nested_model_ftest(self, X_vars_excluded: Iterable):
        """F-test of full model and user-defined smaller model.

        Parameters 
        ----------
        - X_vars_excluded : Iterable. 
            Variables that are excluded in the smaller model. 

        Returns
        -------
        - f_statistic : float
        - p_value : float
        """
        if not self._is_train:
            print(train_only_message)
            return None
        # TODO: implement


    def backward_selection(self, criteria: Literal['r2', 'adjr2', 'aic', 'bic', 
            'mse', 'mad']):
        """Treats the given model as the full model. Iteratively removes 
        """
        if not self._is_train:
            print(train_only_message)
            return None
        # TODO: implement




    def set_outlier_threshold(self, threshold: float):
        """Standardized residuals threshold for outlier identification. 
        Recomputes the outliers.
        
        Parameters
        ----------
        - threshold: float. Must be a nonnegative value.
        """
        if threshold < 0:
            raise ValueError(
                f'Input threshold must be nonnegative. Received {threshold}.')
        self._outlier_threshold = threshold
        self._compute_outliers()


    def get_outlier_indices(self):
        """Returns the indices corresponding to DataFrame examples associated
        with standardized residual outliers. 

        Returns
        -------
        - outliers_df_idx : list ~ (n_outliers)
        """
        return self._outliers_df_idx.tolist()
    



    def fit_statistics(self):
        return self._scorer.to_df()



    def _compute_outliers(self):
        """Computes the outliers. 
        """
        self._outliers_residual_mask =\
            ((self._stdresiduals >= self._outlier_threshold) |
            (self._stdresiduals <= -self._outlier_threshold))
        self._outliers_df_idx = self._X_eval_df.iloc[
            self._outliers_residual_mask].index.to_numpy()


