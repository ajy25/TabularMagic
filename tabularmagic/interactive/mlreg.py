import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable
from ..metrics.regression_scoring import RegressionScorer
from ..ml import *
from ..preprocessing.datapreprocessor import BaseSingleVarScaler


class MLRegressionReport():
    """MLRegressionReport: generates regression-relevant plots and tables 
    for a single machine learning model. 
    """

    def __init__(self, model: BaseRegression, X_test: pd.DataFrame, 
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
        self._y_pred = model.predict(self._X_test_df.to_numpy())
        self._y_true = self._y_test_df.to_numpy().flatten()

        if y_scaler is not None:
            self._y_pred = y_scaler.inverse_transform(self._y_pred)
            self._y_true = y_scaler.inverse_transform(self._y_true)

        self.scorer = RegressionScorer(y_pred=self._y_pred, y_true=self._y_true, 
            n_regressors=model._n_regressors, model_id_str=str(model))
        
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



class ComprehensiveMLRegressionReport():
    """An object that generates regression-relevant plots and tables for a 
    set of models. Indexable. 
    """

    def __init__(self, models: Iterable[BaseRegression], X_test: pd.DataFrame, 
                 y_test: pd.DataFrame, y_scaler: BaseSingleVarScaler = None):
        """
        Initializes a MLRegressionReport object. 

        Parameters 
        ----------
        - models : Iterable[BaseRegression].
            The BaseRegression models must already be trained. 
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
        self.models = models
        self._X_test = X_test
        self._y_test = y_test
        self._report_dict_indexable_by_int = {
            i: MLRegressionReport(model, X_test, y_test, y_scaler) \
                for i, model in enumerate(self.models)}
        self._report_dict_indexable_by_str = {
            str(report.model): report for report in \
                self._report_dict_indexable_by_int.values()
        }
        self.fit_statistics = pd.concat([report.scorer.to_df() for report \
            in self._report_dict_indexable_by_int.values()], axis=1)

    def __getitem__(self, index: int | str) -> MLRegressionReport:
        """Indexes into ComprehensiveMLRegressionReport. 

        Parameters
        ----------
        - index : int | str. 

        Returns
        -------
        - MLRegressionReport. 
        """
        if isinstance(index, int):
            return self._report_dict_indexable_by_int[index]
        elif isinstance(index, str):
            return self._report_dict_indexable_by_str[index]
        else:
            raise ValueError(f'Invalid input: {index}.')


