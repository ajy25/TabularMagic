import pandas as pd
import numpy as np
import matplotlib.axes as axes
from typing import Iterable
from ...metrics.regression_scoring import RegressionScorer
from ...ml.discriminative.regression.base_regression import BaseRegression
from ...preprocessing.datapreprocessor import BaseSingleVarScaler
from ..visualization import plot_pred_vs_true


class MLRegressionReport():
    """MLRegressionReport: generates regression-relevant plots and tables 
    for a single machine learning model. 
    """

    def __init__(self, model: BaseRegression, X_eval: pd.DataFrame = None, 
                 y_eval: pd.Series = None, 
                 y_scaler: BaseSingleVarScaler = None):
        """
        Initializes a RegressionReport object. 

        Parameters
        ----------
        - model : BaseRegression.
            The model must already be trained.
        - X_eval : pd.DataFrame.
            Default: None. If None, uses the model training results directly. 
        - y_eval : pd.Series.
            Default: None. If None, uses the model training results directly. 
        - y_scaler: BaseSingleVarScaler.
            Default: None. If exists, calls inverse transform on the outputs 
            and on y_eval before computing statistics.

        Returns
        -------
        - None
        """
        self.model = model
        if X_eval is not None and y_eval is not None:
            self._y_pred = model.predict(X_eval.to_numpy())
            self._y_true = y_eval.to_numpy()
            self._scorer = RegressionScorer(y_pred=self._y_pred, 
                y_true=self._y_true, n_regressors=model._n_regressors, 
                model_id_str=str(model))
        else:
            self._y_pred = model._y
            self._y_true = model.predict(model._X)
            self._scorer = model.train_scorer
        if y_scaler is not None:
            self._scorer.rescale(y_scaler)
            self.rescale(y_scaler)


    def rescale(self, y_scaler: BaseSingleVarScaler):
        """
        Inverse scales y values.

        Parameters
        ----------
        - y_scaler: BaseSingleVarScaler.
            Calls inverse transform on the outputs 
            and on y_eval before computing statistics.
        """
        if isinstance(self._y_true, np.ndarray):
            self._y_pred = y_scaler.inverse_transform(self._y_pred)
            self._y_true = y_scaler.inverse_transform(self._y_true)
        elif isinstance(self._y_true, list):
            for i in range(len(self._y_true)):
                self._y_true[i] = y_scaler.inverse_transform(self._y_true[i])
                self._y_pred[i] = y_scaler.inverse_transform(self._y_pred[i])


    def fit_statistics(self):
        return self._scorer.to_df()

        
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




class ComprehensiveMLRegressionReport():
    """An object that generates regression-relevant plots and tables for a 
    set of models. Indexable. 
    """

    def __init__(self, models: Iterable[BaseRegression], 
                 X_eval: pd.DataFrame = None, 
                 y_eval: pd.Series = None, 
                 y_scaler: BaseSingleVarScaler = None):
        """
        Initializes a MLRegressionReport object. 

        Parameters 
        ----------
        - models : Iterable[BaseRegression].
            The BaseRegression models must already be trained. 
        - X_eval : pd.DataFrame.
            Default: None. If None, reports on the training data.
        - y_eval : pd.Series.
            Default: None. If None, reports on the training data.
        - y_scaler: BaseSingleVarScaler.
            Default: None. If exists, calls inverse transform on the outputs 
            and on y_eval before computing statistics.

        Returns
        -------
        - None
        """
        self.models = models
        if X_eval is None and y_eval is None:
            self._report_dict_indexable_by_int = {
                i: MLRegressionReport(model=model, y_scaler=y_scaler) \
                    for i, model in enumerate(self.models)}
        else:
            if not isinstance(y_eval, pd.Series):
                try:
                    if isinstance(y_eval, pd.DataFrame):
                        if y_eval.shape[1] > 1:
                            assert False
                        y_eval = y_eval.iloc[:, 0]
                    else:
                        y_eval = pd.Series(y_eval)
                except:
                    raise ValueError('y_eval must be pd.Series object.')
            self._report_dict_indexable_by_int = {
                i: MLRegressionReport(model, X_eval, y_eval, y_scaler) \
                    for i, model in enumerate(self.models)}
        self._report_dict_indexable_by_str = {
            str(report.model): report for report in \
                self._report_dict_indexable_by_int.values()
        }
        self._fit_statistics = pd.concat([report._scorer.to_df() for report \
            in self._report_dict_indexable_by_int.values()], axis=1)
        

    def fit_statistics(self):
        return self._fit_statistics


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


