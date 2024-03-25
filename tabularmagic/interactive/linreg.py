import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import seaborn as sns
from scipy import stats
from typing import Iterable
from ..metrics.regression_scoring import RegressionScorer
from ..linear import *
from ..preprocessing.datapreprocessor import BaseSingleVarScaler
from .visualization import *



class LinearRegressionReport():
    """LinearRegressionReport: generates regression-relevant diagnostic 
    plots and tables for a single linear regression model. 
    """
    
    def __init__(self, model: OrdinaryLeastSquares, X_eval: pd.DataFrame, 
                 y_eval: pd.DataFrame, y_scaler: BaseSingleVarScaler = None):
        """
        Initializes a RegressionReport object. 

        Parameters x
        ----------
        - model : BaseRegression.
            The model must already be trained.
        - X_eval : pd.DataFrame.
        - y_eval : pd.DataFrame.
        - y_scaler: BaseSingleVarScaler.
            Default: None. If exists, calls inverse transform on the outputs 
            and on y_eval before computing statistics.
        """
        if not isinstance(y_eval, pd.DataFrame):
            y_eval = y_eval.to_frame()
        self.model = model
        self._X_eval_df = X_eval
        self._y_eval_df = y_eval
        self._y_pred = model.predict(self._X_eval_df)
        self._is_train = False
        if len(self._y_pred) == len(model.estimator.fittedvalues):
            if np.allclose(self._y_pred, model.estimator.fittedvalues):
                self._is_train = True
        self._y_true = self._y_eval_df.to_numpy().flatten()
        if y_scaler is not None:
            self._y_pred = y_scaler.inverse_transform(self._y_pred)
            self._y_true = y_scaler.inverse_transform(self._y_true)
        self.scorer = RegressionScorer(y_pred=self._y_pred, y_true=self._y_true, 
            n_regressors=model._n_regressors, model_id_str=str(model))
        self._residuals = self._y_true - self._y_pred
        
        
    def statsmodels_summary(self):
        """
        Returns the summary of the statsmodels RegressionResultsWrapper for 
        OLS.
        """
        try:
            return self.model.estimator.summary()
        except:
            raise RuntimeError('Error occured in statsmodels_summary call.')


    def plot_pred_vs_true(self, figsize: Iterable = (5, 5), 
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
        return plot_pred_vs_true(self._y_pred, self._y_true, figsize, ax)
    

    def plot_residuals_vs_fitted(self, standardized: bool = False, 
                                 figsize: Iterable = (5, 5), 
                                 ax: axes.Axes = None):
        """Returns a figure that is a residuals vs fitted (y_pred) plot. 

        Parameters
        ----------
        - standardized : bool. If True, standardizes the residuals. 
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
            residuals = residuals / residuals.std()
        ax.scatter(self._y_pred, residuals, s=2, color='black')
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
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
                              figsize: Iterable = (5, 5), 
                              ax: axes.Axes = None):
        """Returns a figure that is a residuals vs fitted (y_pred) plot. 
        
        Parameters
        ----------
        - x_var : str.
        - standardized : bool. If True, standardizes the residuals. 
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
            residuals = (residuals - residuals.mean()) / residuals.std()
        ax.scatter(self._X_eval_df[x_var].to_numpy(), residuals, s=2, 
                   color='black')
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
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
            residuals = residuals / residuals.std()
        sns.histplot(residuals, bins='auto', color='black', 
                     edgecolor='black', 
                     stat=stat, ax=ax, kde=True)
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
    

    def plot_scale_location(self, figsize: Iterable = (5, 5), 
                            ax: axes.Axes = None):
        """Returns a figure that is a plot of the 
        sqrt of the residuals versus the fitted.
        
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
        
        residuals = self._residuals
        residuals = residuals / residuals.std()

        ax.scatter(self._y_pred, np.sqrt(residuals), s=2, 
            color='black')
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax.set_xlabel('Fitted')
        ax.set_ylabel('sqrt(Standardized Residuals)')
        ax.set_title(f'Scale-Location')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
        if fig is not None:
            fig.tight_layout()
            plt.close()
        return fig

    def plot_residuals_vs_leverage(self, standardized: bool = True, 
                                   figsize: Iterable = (5, 5), 
                                   ax: axes.Axes = None):
        """Returns a figure that is a plot of the residuals versus leverage.
        
        Parameters
        ----------
        - standardized : bool. If True, standardizes the residuals. 
        - figsize : Iterable.
        - ax : axes.Axes. 

        Returns
        -------
        - plt.Figure
        """
        if not self._is_train:
            return

        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        leverage = self.model.estimator._results.get_influence().hat_matrix_diag
        residuals = self._residuals
        if standardized:
            residuals = residuals / residuals.std()
        ax.scatter(leverage, residuals, s=2, 
                   color='black')
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
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
    

    
    def plot_qq(self, figsize: Iterable = (5, 5), ax: axes.Axes = None):
        """Returns a quantile-quantile plot.
        
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


        tup1, _ = stats.probplot(self._residuals / self._residuals.std(), 
                       dist='norm')
        
        thoeretical_quantitles, std_res = tup1
        

        ax.set_title('Q-Q Plot')
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Standardized Residuals')

        temp_stack = np.hstack((thoeretical_quantitles, std_res))
        min_val = np.min(temp_stack)
        max_val = np.max(temp_stack)
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
        ax.scatter(thoeretical_quantitles, std_res, s=2, color='black', 
                   marker='o')

        if fig is not None:
            fig.tight_layout()
            plt.close()
        return fig


    def plot_diagnostics(self, figsize: Iterable = (7, 7)):
        """Plots several useful linear regression diagnostic plots.

        Parameters 
        ----------
        - figsize : Iterable.

        Returns
        -------
        - plt.Figure
        """
        fig, axs = plt.subplots(2, 2, figsize=figsize)

        self.plot_pred_vs_true(ax=axs[0][0])
        self.plot_residuals_vs_fitted(ax=axs[0][1])

        if self._is_train:
            self.plot_residuals_vs_leverage(ax=axs[1][0])
        else:
            self.plot_scale_location(ax=axs[1][0])

        self.plot_qq(ax=axs[1][1])
        
        fig.subplots_adjust(hspace=0.25, wspace=0.25)

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
        pass






    

