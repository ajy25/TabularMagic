import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable
from ..metrics.regression_scoring import RegressionScorer
from ..linear import *
from ..preprocessing.datapreprocessor import BaseSingleVarScaler


class LinearRegressionReport():
    """LinearRegressionReport: generates regression-relevant plots and 
    tables for a single OLS model. 
    """
    
    def __init__(self, model: OrdinaryLeastSquares, X_test: pd.DataFrame, 
                 y_test: pd.DataFrame, y_scaler: BaseSingleVarScaler = None):
        """
        Initializes a RegressionReport object. 

        Parameters x
        ----------
        - model : BaseRegression.
            The model must already be trained.
        - X_test : pd.DataFrame.
        - y_test : pd.DataFrame.
        - y_scaler: BaseSingleVarScaler.
            Default: None. If exists, calls inverse transform on the outputs 
            and on y_test before computing statistics.

        Returns
        -------
        - None
        """
        if not isinstance(y_test, pd.DataFrame):
            y_test = y_test.to_frame()
        self.model = model
        self._X_test_df = X_test
        self._y_test_df = y_test
        self._y_pred = model.predict(self._X_test_df)
        self._y_true = self._y_test_df.to_numpy().flatten()
        if y_scaler is not None:
            self._y_pred = y_scaler.inverse_transform(self._y_pred)
            self._y_true = y_scaler.inverse_transform(self._y_true)
        self.scorer = RegressionScorer(y_pred=self._y_pred, y_true=self._y_true, 
            n_regressors=model._n_regressors, model_id_str=str(model))
        
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
        - figsize: Iterable

        Returns
        -------
        - plt.Figure
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.scatter(self._y_true, self._y_pred, s=2, color='black')
        min_val = np.min(np.hstack((self._y_pred, self._y_true)))
        max_val = np.max(np.hstack((self._y_pred, self._y_true)))
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
        ax.set_title(f'{self._y_test_df.columns.to_list()[0]} | ' + \
                     f'ρ = {round(self.scorer["pearsonr"], 3)}')
        fig.tight_layout()
        return fig




