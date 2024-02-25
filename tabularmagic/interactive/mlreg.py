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

    def __init__(self, model: BaseRegression, X_test: pd.DataFrame = None, 
                 y_test: pd.DataFrame = None, 
                 y_scaler: BaseSingleVarScaler = None):
        """
        Initializes a RegressionReport object. 

        Parameters
        ----------
        - model : BaseRegression.
            The model must already be trained.
        - X_test : pd.DataFrame.
            Default: None. If None, uses the model training results directly. 
        - y_test : pd.DataFrame.
            Default: None. If None, uses the model training results directly. 
        - y_scaler: BaseSingleVarScaler.
            Default: None. If exists, calls inverse transform on the outputs 
            and on y_test before computing statistics.

        Returns
        -------
        - None
        """
        self.model = model
        if X_test is not None and y_test is not None:
            self._y_pred = model.predict(X_test.to_numpy())
            self._y_true = y_test.to_numpy().flatten()
            self.scorer = RegressionScorer(y_pred=self._y_pred, 
                y_true=self._y_true, n_regressors=model._n_regressors, 
                model_id_str=str(model))
        else:
            self._y_pred = model._y
            self._y_true = model.predict(model._X)
            self.scorer = model.train_scorer
        if y_scaler is not None:
            self.scorer.rescale(y_scaler)
            self.rescale(y_scaler)


    def rescale(self, y_scaler: BaseSingleVarScaler):
        """
        Inverse scales y values.

        Parameters
        ----------
        - y_scaler: BaseSingleVarScaler.
            Calls inverse transform on the outputs 
            and on y_test before computing statistics.
        """
        if isinstance(self._y_true, np.ndarray):
            self._y_pred = y_scaler.inverse_transform(self._y_pred)
            self._y_true = y_scaler.inverse_transform(self._y_true)
        elif isinstance(self._y_true, list):
            for i in range(len(self._y_true)):
                self._y_true[i] = y_scaler.inverse_transform(self._y_true[i])
                self._y_pred[i] = y_scaler.inverse_transform(self._y_pred[i])

        
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
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
        ax.set_title(f'Predicted vs True | ' + \
                     f'ρ = {round(self.scorer["pearsonr"], 3)}')
        ax.ticklabel_format(style='sci', axis='both', scilimits=(-2, 2))
        fig.tight_layout()
        plt.close(fig)
        return fig



class ComprehensiveMLRegressionReport():
    """An object that generates regression-relevant plots and tables for a 
    set of models. Indexable. 
    """

    def __init__(self, models: Iterable[BaseRegression], 
                 X_test: pd.DataFrame = None, 
                 y_test: pd.DataFrame = None, 
                 y_scaler: BaseSingleVarScaler = None):
        """
        Initializes a MLRegressionReport object. 

        Parameters 
        ----------
        - models : Iterable[BaseRegression].
            The BaseRegression models must already be trained. 
        - X_test : pd.DataFrame.
            Default: None. If None, reports on the training data.
        - y_test : pd.DataFrame.
            Default: None. If None, reports on the training data.
        - y_scaler: BaseSingleVarScaler.
            Default: None. If exists, calls inverse transform on the outputs 
            and on y_test before computing statistics.

        Returns
        -------
        - None
        """
        self.models = models
        if X_test is None and y_test is None:
            self._report_dict_indexable_by_int = {
                i: MLRegressionReport(model=model, y_scaler=y_scaler) \
                    for i, model in enumerate(self.models)}
        else:
            if not isinstance(y_test, pd.DataFrame):
                y_test = y_test.to_frame()
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


