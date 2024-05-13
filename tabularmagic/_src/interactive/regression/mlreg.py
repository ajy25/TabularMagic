import pandas as pd
import matplotlib.axes as axes
import matplotlib.figure as figure
from typing import Iterable, Literal
from ...ml.discriminative.regression.base_regression import BaseRegression
from ...data.datahandler import DataHandler
from ..visualization import plot_obs_vs_pred
from ...util.console import print_wrapped





class SingleModelSingleDatasetMLRegReport:
    """
    SingleModelSingleDatasetMLReport: generates regression-relevant plots and
    tables for a single machine learning model on a single dataset.
    """

    def __init__(self, model: BaseRegression, 
                 dataset: Literal['train', 'test']):
        """
        Initializes a SingleModelSingleDatasetMLReport object.
        
        Parameters
        ----------
        - model : BaseRegression. The data for the model must already be 
            specified. The model should already be trained on the 
            specified data.
        - dataset : Literal['train', 'test'].
        """
        self.model = model
        if dataset not in ['train', 'test']:
            raise ValueError('dataset must be either "train" or "test".')
        self._dataset = dataset

    def fit_statistics(self) -> pd.DataFrame:
        """Returns a DataFrame containing the goodness-of-fit statistics
        for the model on the specified data.

        Parameters
        ----------
        - pd.DataFrame
        """
        if self._dataset == 'train':
            return self.model.train_scorer.stats_df()
        else:
            return self.model.test_scorer.stats_df()
        
    def cv_fit_statistics(self) -> pd.DataFrame:
        """Returns a DataFrame containing the cross-validated goodness-of-fit 
        statistics for the model on the specified data.

        Parameters
        ----------
        - pd.DataFrame
        """
        if self._dataset == 'train':
            return self.model.train_scorer.cv_df()
        else:
            raise ValueError(
                'Cross-validated statistics are not available for test data.')
    
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
        if self._dataset == 'train':
            if self.model.train_overall_scorer is not None:
                # in case of nested cross validation, use overall scorer
                y_pred = self.model.train_overall_scorer._y_pred
                y_true = self.model.train_overall_scorer._y_true
            else:
                y_pred = self.model.train_scorer._y_pred
                y_true = self.model.train_scorer._y_true
        else:
            y_pred = self.model.test_scorer._y_pred
            y_true = self.model.test_scorer._y_true
        return plot_obs_vs_pred(y_pred, y_true, figsize, ax)


class SingleModelMLRegReport:
    """SingleModelMLRegReport: generates regression-relevant plots and 
    tables for a single machine learning model. 
    """

    def __init__(self, model: BaseRegression):
        """
        Initializes a SingleModelMLRegReport object. 

        Parameters
        ----------
        - model : BaseRegression. The data for the model must already be 
            specified. The model should already be trained on the 
            specified data.
        """
        self.model = model


    def train_report(self) -> SingleModelSingleDatasetMLRegReport:
        """Returns a SingleModelSingleDatasetMLReport object for the training data.

        Returns
        -------
        - SingleModelSingleDatasetMLReport
        """
        return SingleModelSingleDatasetMLRegReport(self.model, 'train')
    
    def test_report(self) -> SingleModelSingleDatasetMLRegReport:
        """Returns a SingleModelSingleDatasetMLReport object for the test data.

        Returns
        -------
        - SingleModelSingleDatasetMLReport
        """
        return SingleModelSingleDatasetMLRegReport(self.model, 'test')





class MLRegressionReport:
    """MLRegressionReport.  
    Fits the model based on provided DataHandler.
    Wraps train and test SingleDatasetMLRegReport objects.
    """

    def __init__(self, models: Iterable[BaseRegression], 
                 datahandler: DataHandler,
                 y_var: str,
                 X_vars: list[str],
                 outer_cv: int = None,
                 outer_cv_seed: int = 42,
                 verbose: bool = True):
        """MLRegressionReport.  
        Fits the model based on provided DataHandler.
        Wraps train and test SingleDatasetMLRegReport objects.
        
        Parameters 
        ----------
        - models : Iterable[BaseRegression].
            The BaseRegression models must already be trained. 
        - datahandler : DataHandler.
        - y_var : str.
        - X_vars : list[str].
        - outer_cv : int.
            If not None, reports training scores via nested k-fold CV.
        - outer_cv_seed : int.
            The random seed for the outer cross validation loop.
        - verbose : bool.
        """
        self._models = models
        self._id_to_model = {model._name: model for model in models}


        self._datahandler = datahandler
        self._X_vars = X_vars
        self._y_var = y_var
        self._verbose = verbose
        for model in self._models:
            if self._verbose:
                print_wrapped(f'Fitting model {model._name}.', 
                            type='UPDATE')
            model.specify_data(datahandler, y_var, X_vars, outer_cv, 
                                outer_cv_seed)
            model.fit()
            if self._verbose:
                print_wrapped(f'Fitted model {model._name}.', 
                            type='UPDATE')
        
        self._id_to_report = {model._name: SingleModelMLRegReport(model) \
                              for model in models}
    

    def get_model_report(self, model_id: str) -> SingleModelMLRegReport:
        """Returns the SingleModelMLRegReport object for the specified model.

        Parameters
        ----------
        - model_id : str. The id of the model.

        Returns
        -------
        - SingleModelMLRegReport
        """
        return self._id_to_report[model_id]
    
    def get_model(self, model_id: str) -> BaseRegression:
        """Returns the model with the specified id.

        Parameters
        ----------
        - model_id : str. The id of the model.

        Returns
        -------
        - BaseRegression
        """
        return self._id_to_model[model_id]
    

    def fit_statistics(self,
                       dataset: Literal['train', 'test']) -> pd.DataFrame:
        """Returns a DataFrame containing the goodness-of-fit statistics for 
        all models on the specified data.
        
        Parameters
        ----------
        - dataset : Literal['train', 'test'].

        Returns
        -------
        - pd.DataFrame
        """
        if dataset == 'train':
            return pd.concat([report.train_report().fit_statistics() \
                              for report in self._id_to_report.values()], 
                              axis=1)
        else:
            return pd.concat([report.test_report().fit_statistics() \
                              for report in self._id_to_report.values()], 
                              axis=1)

    
    def __getitem__(self, model_id: str) -> SingleModelMLRegReport:
        return self._id_to_report[model_id]



