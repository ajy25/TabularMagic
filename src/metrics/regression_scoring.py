import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy.stats import (
    pearsonr, spearmanr
)
from typing import Literal



class RegressionScorer():
    """RegressionScorer : Class built for simple scoring of regression fits.

    RegressionScorer is indexable by integers in the following order: 
        (MSE, MAD, Pearson Correlation, Spearman Correlation, R Squared, 
        Adjusted R Squared). 

    RegressionScorer also is indexable by a string key similar to the 
    dictionary: {'mse': MSE, 'mad': MAD, 'pearsonr': Pearson Correlation, 
        'spearmanr': Spearman Correlation, 'r2': R Squared, 
        'adjr2': Adjusted R Squared}. 
    """

    def __init__(self, y_pred: np.ndarray, y_true: np.ndarray, 
                 n_regressors: int = None, model_id_str: str = None):
        """
        Initializes a RegressionScorer object. 

        Parameters
        ----------
        - y_pred : np.ndarray ~ (n_samples).
        - y_true : np.ndarray ~ (n_samples).

        Returns
        -------
        - None
        """
        if model_id_str is None:
            self._model_id_str = 'Model'
        else:
            self._model_id_str = model_id_str
        self._y_pred = y_pred.copy()
        self._y_true = y_true.copy()
        self._pearsonr = pearsonr(self._y_pred, self._y_true)[0]
        self._spearmanr = spearmanr(self._y_pred, self._y_true)[0]
        self._mse = mean_squared_error(self._y_pred, self._y_true)
        self._mad = mean_absolute_error(self._y_pred, self._y_true)
        self._rsquared = r2_score(self._y_pred, self._y_true)
        self.n_regressors = n_regressors
        self.n_samples = self._y_pred.shape[0]
        if n_regressors is None:
            self._adjustedrsquared = np.NaN
        else: 
            self._adjustedrsquared = 1 - (((1 - self._rsquared) * \
                (self.n_samples - 1)) / (self.n_samples - self.n_regressors \
                - 1))
        self._dict_indexable_by_str = {
            'mse': self._mse,
            'mad': self._mad,
            'pearsonr': self._pearsonr,
            'spearmanr': self._spearmanr,
            'r2': self._rsquared,
            'adjr2': self._adjustedrsquared
        }
        self._dict_indexable_by_int = {
            0: self._mse,
            1: self._mad,
            2: self._pearsonr,
            3: self._spearmanr,
            4: self._rsquared,
            5: self._adjustedrsquared
        }

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

    
