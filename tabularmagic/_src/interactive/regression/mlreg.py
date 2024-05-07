import pandas as pd
import numpy as np
import matplotlib.axes as axes
import matplotlib.figure as figure
from typing import Iterable
from ...metrics.regression_scoring import RegressionScorer
from ...ml.discriminative.regression.base_regression import BaseRegression
from ...data.preprocessing import BaseSingleVarScaler
from ..visualization import plot_obs_vs_pred


class MLRegressionReportSingleModel:
    """MLRegressionReportSingleModel: generates regression-relevant plots and 
    tables for a single machine learning model. 
    """

    def __init__(self, model: BaseRegression, X_test: pd.DataFrame = None, 
                 y_test: pd.Series = None, 
                 y_scaler: BaseSingleVarScaler = None):
        """
        Initializes a MLRegressionReportSingleModel object. 

        Parameters
        ----------
        - model : BaseRegression.
            The model must already be trained.
        - X_test : pd.DataFrame.
            Default: None. If None, uses the model training results directly. 
        - y_test : pd.Series.
            Default: None. If None, uses the model training results directly. 
        - y_scaler: BaseSingleVarScaler.
            Default: None. If exists, calls inverse transform on the outputs 
            and on y_test before computing statistics.
        """
        self.model = model
        if X_test is not None and y_test is not None:
            self._y_pred = model.predict(X_test.to_numpy())
            self._y_true = y_test.to_numpy()
            self._scorer = RegressionScorer(y_pred=self._y_pred, 
                y_true=self._y_true, n_predictors=model._n_predictors, 
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
            and on y_test before computing statistics.
        """
        if isinstance(self._y_true, np.ndarray):
            self._y_pred = y_scaler.inverse_transform(self._y_pred)
            self._y_true = y_scaler.inverse_transform(self._y_true)
        elif isinstance(self._y_true, list):
            for i in range(len(self._y_true)):
                self._y_true[i] = y_scaler.inverse_transform(self._y_true[i])
                self._y_pred[i] = y_scaler.inverse_transform(self._y_pred[i])


    def fit_statistics(self) -> pd.DataFrame:
        """Returns a DataFrame containing the goodness-of-fit statistics
        for the model.

        Parameters
        ----------
        - pd.DataFrame
        """
        return self._scorer.to_df()

        
    def plot_obs_vs_pred(self, figsize: Iterable = (5, 5), 
                          ax: axes.Axes = None) -> figure.Figure:
        """Returns a figure that is a scatter plot of the observed (y-axis) and 
        predicted (x-axis) values. 

        Parameters 
        ----------
        - figsize : Iterable
        - ax : Axes

        Returns
        -------
        - Figure
        """
        return plot_obs_vs_pred(self._y_pred, self._y_true, figsize, ax)




class MLRegressionReport:
    """An object that generates regression-relevant plots and tables for a 
    set of models. Indexable. 
    """

    def __init__(self, models: Iterable[BaseRegression], 
                 X_test: pd.DataFrame = None, 
                 y_test: pd.Series = None, 
                 y_scaler: BaseSingleVarScaler = None):
        """
        Initializes a MLRegressionReport object. 

        Parameters 
        ----------
        - models : Iterable[BaseRegression].
            The BaseRegression models must already be trained. 
        - X_test : pd.DataFrame.
            Default: None. If None, reports on the training data.
        - y_test : pd.Series.
            Default: None. If None, reports on the training data.
        - y_scaler: BaseSingleVarScaler.
            Default: None. If exists, calls inverse transform on the outputs 
            and on y_test before computing statistics.
        """
        self.models = models
        if X_test is None and y_test is None:
            self._report_dict_indexable_by_int = {
                i: MLRegressionReportSingleModel(model=model, 
                                                 y_scaler=y_scaler) \
                    for i, model in enumerate(self.models)}
        else:
            if not isinstance(y_test, pd.Series):
                try:
                    if isinstance(y_test, pd.DataFrame):
                        if y_test.shape[1] > 1:
                            assert False
                        y_test = y_test.iloc[:, 0]
                    else:
                        y_test = pd.Series(y_test)
                except:
                    raise ValueError('y_test must be pd.Series object.')
            self._report_dict_indexable_by_int = {
                i: MLRegressionReportSingleModel(model, X_test, y_test, 
                                                 y_scaler) \
                    for i, model in enumerate(self.models)}
        self._report_dict_indexable_by_str = {
            str(report.model): report for report in \
                self._report_dict_indexable_by_int.values()
        }
        self._fit_statistics = pd.concat([report._scorer.to_df() for report \
            in self._report_dict_indexable_by_int.values()], axis=1)
        

    def fit_statistics(self):
        return self._fit_statistics


    def __getitem__(self, index: int | str) -> MLRegressionReportSingleModel:
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



class ComprehensiveMLRegressionReport:
    """ComprehensiveMLRegressionReport: Simple object that combines 
    the MLRegressionReports for both train and test."""

    def __init__(self, models: Iterable[BaseRegression], 
                 X_test: pd.DataFrame, 
                 y_test: pd.Series, 
                 y_scaler: BaseSingleVarScaler = None):
        """MLRegressionReport wrapper for both train and test
        
        Parameters 
        ----------
        - models : Iterable[BaseRegression].
            The BaseRegression models must already be trained. 
        - X_test : pd.DataFrame.
        - y_test : pd.Series.
        - y_scaler: BaseSingleVarScaler.
            Default: None. If exists, calls inverse transform on the outputs 
            and on y_test before computing statistics.
        """
        self._train_report = MLRegressionReport(
            models, y_scaler=y_scaler)
        self._test_report = MLRegressionReport(
            models, X_test, y_test, y_scaler)
        
    def train(self):
        """Returns an MLRegressionReport object for the train dataset
        
        Returns
        -------
        - report : MLRegressionReport
        """
        return self._train_report
    
    def test(self):
        """Returns an MLRegressionReport object for the test dataset
        
        Returns
        -------
        - report : MLRegressionReport
        """
        return self._test_report




