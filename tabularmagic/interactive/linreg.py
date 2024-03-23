import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable
from ..metrics.regression_scoring import RegressionScorer
from ..linear import *
from ..preprocessing.datapreprocessor import BaseSingleVarScaler
from .visualization import plot_pred_vs_true
import seaborn as sns


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

        Returns
        -------
        - None
        """
        if not isinstance(y_eval, pd.DataFrame):
            y_eval = y_eval.to_frame()
        self.model = model
        self._X_eval_df = X_eval
        self._y_eval_df = y_eval
        self._y_pred = model.predict(self._X_eval_df)
        self._y_true = self._y_eval_df.to_numpy().flatten()
        if y_scaler is not None:
            self._y_pred = y_scaler.inverse_transform(self._y_pred)
            self._y_true = y_scaler.inverse_transform(self._y_true)
        self.scorer = RegressionScorer(y_pred=self._y_pred, y_true=self._y_true, 
            n_regressors=model._n_regressors, model_id_str=str(model))
        self._residuals = self._y_true - self._y_pred
        
        
    def statsmodels_summary(self):
        try:
            return self.model.estimator.summary()
        except:
            raise RuntimeError('Error occured in statsmodels_summary call.')


    def plot_pred_vs_true(self, figsize: Iterable = (5, 5)):
        """Returns a figure that is a scatter plot of the true and predicted y 
        values. 

        Parameters 
        ----------
        - figsize : Iterable. 

        Returns
        -------
        - plt.Figure
        """
        return plot_pred_vs_true(self._y_pred, self._y_true, figsize)
    

    def plot_residuals_vs_fitted(self, standardized: bool = False, 
                                 figsize: Iterable = (5, 5)):
        """Returns a figure that is a residuals vs fitted (y_pred) plot. 

        Parameters
        ----------
        - figsize : Iterable.

        Returns
        -------
        - plt.Figure
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        residuals = self._residuals
        if standardized:
            residuals = (residuals - residuals.mean()) / residuals.std()
        ax.scatter(self._y_pred, residuals, s=2, color='black')
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Fitted')
        if standardized:
            ax.set_ylabel('Standardized Residuals')
            ax.set_title(f'Standardized Residuals vs Fitted')
        else:
            ax.set_ylabel('Residuals')
            ax.set_title(f'Residuals vs Fitted')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
        fig.tight_layout()
        plt.close()
        return fig


    def plot_residuals_vs_var(self, x_var: str, standardized: bool = False, 
                              figsize: Iterable = (5, 5)):
        """Returns a figure that is a residuals vs fitted (y_pred) plot. 
        
        Parameters
        ----------
        - x_var : str.
        - standardized : bool. If True, standardizes the residuals. 
        - figsize : Iterable.

        Returns
        -------
        - plt.Figure
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        residuals = self._residuals
        if standardized:
            residuals = (residuals - residuals.mean()) / residuals.std()
        ax.scatter(self._X_eval_df[x_var].to_numpy(), residuals, s=2, 
                   color='black')
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel(x_var)
        if standardized:
            ax.set_ylabel('Standardized Residuals')
            ax.set_title(f'Standardized Residuals vs {x_var}')
        else:
            ax.set_ylabel('Residuals')
            ax.set_title(f'Residuals vs {x_var}')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, 3))
        fig.tight_layout()
        plt.close()
        return fig
    
    def plot_residuals_hist(self, standardized: bool = False,
                            density: bool = False, 
                            figsize: Iterable = (5, 5)):
        """Returns a figure that is a histogram of the residuals.

        Parameters
        ----------
        - standardized : bool. If True, standardizes the residuals. 
        - density : bool. If True, plots density rather than frequency.
        - figsize : Iterable.

        Returns
        -------
        - plt.Figure
        """
        if density:
            stat = 'density'
        else:
            stat = 'count'
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        residuals = self._residuals
        if standardized:
            residuals = (residuals - residuals.mean()) / residuals.std()
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
        fig.tight_layout()
        plt.close()
        return fig
    

    def plot_residuals_vs_leverage(self, figsize: Iterable = (5, 5)):
        """Returns a figure that is a plot of the residuals versus leverage.
        
        Parameters
        ----------
        - figsize : Iterable.

        Returns
        -------
        - plt.Figure
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    
    def plot_qq(self, figsize: Iterable = (5, 5)):
        """Returns a quantile-quantile plot.
        
        Parameters 
        ----------
        - figsize : Iterable.

        Returns
        -------
        - plt.Figure
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)


    def plot_diagnostics(self, figsize: Iterable = (5, 5)):
        """Plots several useful linear regression diagnostic plots.

        Parameters 
        ----------
        - figsize : Iterable.

        Returns
        -------
        - plt.Figure
        """
        fig, ax = plt.subplots(2, 2, figsize=figsize)




    

    def nested_model_ftest(self, X_vars_excluded: Iterable):
        """F-test of full model and user-defined smaller model.

        Parameters 
        ----------
        - figsize : Iterable.

        Returns
        -------
        - f_statistic : float
        - p_value : float
        """
        pass






    

