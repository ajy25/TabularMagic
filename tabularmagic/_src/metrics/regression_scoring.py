import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy.stats import (
    pearsonr, spearmanr
)
from typing import Literal
from ..preprocessing.datapreprocessor import BaseSingleVarScaler



class RegressionScorer():
    """RegressionScorer : Class built for simple scoring of regression fits.
    Only inputs are predicted and true values.
    Capable of scoring cross validation outputs.

    RegressionScorer is indexable by integers in the following order: 
        (MSE, MAD, Pearson Correlation, Spearman Correlation, R Squared, 
        Adjusted R Squared). 

    RegressionScorer also is indexable by a string key similar to the 
    dictionary: {'mse': MSE, 'mad': MAD, 'pearsonr': Pearson Correlation, 
        'spearmanr': Spearman Correlation, 'r2': R Squared, 
        'adjr2': Adjusted R Squared}. 
    """

    def __init__(self, y_pred: np.ndarray | list, y_true: np.ndarray | list, 
                 n_predictors: int = None, model_id_str: str = None):
        """
        Initializes a RegressionScorer object. 

        Parameters
        ----------
        - y_pred : np.ndarray ~ (n_samples) | list[np.ndarray ~ (n_samples)].
        - y_true : np.ndarray ~ (n_samples) | list[np.ndarray ~ (n_samples)].
        - n_predictors : int.
        - model_id_str : str.

        Returns
        -------
        - None
        """
        if model_id_str is None:
            self._model_id_str = 'Model'
        else:
            self._model_id_str = model_id_str
        self.n_predictors = n_predictors
        self._y_pred = y_pred
        self._y_true = y_true
        self._dict_indexable_by_str = self._compute_stats_dict(y_pred, y_true)
        self._dict_indexable_by_int = {i: value for i, (_, value) in \
            enumerate(self._dict_indexable_by_str.items())}
        self.cv_metrics = None


    def _compute_stats_dict(self, y_pred: np.ndarray | list, 
                            y_true: np.ndarray | list):
        """
        Returns a statistics dictionary given y_pred and y_true. If y_pred and
        y_true are lists, then the elements are treated as cross 
        validation folds, and the statistics are averaged across all 
        folds.

        Parameters
        ----------
        - y_pred : np.ndarray ~ (n_samples) | list[np.ndarray ~ (n_samples)].
        - y_true : np.ndarray ~ (n_samples) | list[np.ndarray ~ (n_samples)].

        Returns
        -------
        - dict ~ {statistic (str) : value (float)} | 
            [{statistic (str) : value (float)}]
        """
        if isinstance(y_pred, np.ndarray) and isinstance(y_true, np.ndarray):
            n = len(y_pred)
            metrics_dict = dict()
            metrics_dict['mse'] = mean_squared_error(y_true, y_pred)
            metrics_dict['mad'] = mean_absolute_error(y_true, y_pred)
            metrics_dict['pearsonr'] = pearsonr(y_true, y_pred)[0]
            metrics_dict['spearmanr'] = spearmanr(y_true, y_pred)[0]
            metrics_dict['r2'] = r2_score(y_true, y_pred)
            if self.n_predictors is None:
                metrics_dict['adjr2'] = np.NaN
            else: 
                metrics_dict['adjr2'] = 1 - (((1 - metrics_dict['r2']) * \
                    (n - 1)) / (n - \
                    self.n_predictors - 1))
            metrics_dict['n'] = len(y_true)
            output = metrics_dict
        elif isinstance(y_pred, list) and isinstance(y_true, list):
            assert len(y_pred) == len(y_true)
            n_folds = len(y_true)
            folds_metrics = []
            for y_pred_elem, y_true_elem in zip(y_pred, y_true):
                n = len(y_pred_elem)
                metrics_dict = dict()
                metrics_dict['mse'] =\
                    mean_squared_error(y_true_elem, y_pred_elem)
                metrics_dict['mad'] =\
                    mean_absolute_error(y_true_elem, y_pred_elem)
                metrics_dict['pearsonr'] =\
                    pearsonr(y_true_elem, y_pred_elem)[0]
                metrics_dict['spearmanr'] =\
                    spearmanr(y_true_elem, y_pred_elem)[0]
                metrics_dict['r2'] = r2_score(y_true_elem, y_pred_elem)
                if self.n_predictors is None:
                    metrics_dict['adjr2'] = np.NaN
                else: 
                    metrics_dict['adjr2'] = 1 - (((1 - metrics_dict['r2']) * \
                        (n - 1)) / (n - \
                        self.n_predictors - 1))
                metrics_dict['n'] = n
                folds_metrics.append(metrics_dict)
            self.cv_metrics = metrics_dict

            output = dict()
            for fold in folds_metrics:
                for key in fold.keys():
                    if key not in output:
                        output[key] = fold[key] / n_folds
                    else:
                        output[key] += fold[key] / n_folds
        else:
            raise ValueError('Input types for y_pred and y_true are invalid.')
        return output
    

    def rescale(self, y_scaler: BaseSingleVarScaler):
        """
        Inverse scales y values, then recomputes fit statistics.

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
        self._dict_indexable_by_str = self._compute_stats_dict(self._y_pred, 
                                                               self._y_true)
        self._dict_indexable_by_int = {i: value for i, (_, value) in \
            enumerate(self._dict_indexable_by_str.items())}
    

    def __getitem__(self, index: Literal[0, 1, 2, 3, 4, 5] | \
            Literal['mse', 'mad', 'pearsonr', 'spearmanr', 'r2', 'adjr2']):
        """
        Indexes into RegressionScorer. 
        
        RegressionScorer is indexable by integers in the following order: 
            (MSE, MAD, Pearson Correlation, Spearman Correlation, R Squared, 
            Adjusted R Squared). 

        RegressionScorer also is indexable by a string key similar to the 
        dictionary: {'mse': MSE, 'mad': MAD, 'pearsonr': Pearson Correlation, 
            'spearmanr': Spearman Correlation, 'r2': R Squared, 
            'adjr2': Adjusted R Squared}. 

        Parameters
        ----------
        - index : int | str. 

        Returns
        -------
        - float. Value of the appropriate statistic. 
        """
        if isinstance(index, int):
            return self._dict_indexable_by_int[index]
        elif isinstance(index, str):
            return self._dict_indexable_by_str[index]
        else:
            raise ValueError(f'Invalid input: {index}.')
        
    def to_df(self):
        """Outputs a DataFrame that contains all the statistics.

        Parameters
        ----------
        - None

        Returns
        -------
        - pd.DataFrame.
        """
        return pd.DataFrame(list(self._dict_indexable_by_str.items()), 
            columns=['Statistic', self._model_id_str]).set_index('Statistic')

    
