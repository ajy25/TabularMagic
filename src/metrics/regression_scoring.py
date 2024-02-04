import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy.stats import (
    pearsonr, spearmanr
)
from typing import Literal



class RegressionScorer():
    """RegressionScorer : Class built for simple scoring of regression fits.
    Indexable by integer in the following order: 
    (MSE, MAD, Pearson Correlation, Spearman Correlation, R Squared). 
    Indexable by string key similar to the following dictionary:
    {'mse': MSE, 'mad': MAD, 'pearsonr': Pearson Correlation, 'spearmanr': 
    Spearman Correlation, 'r2': R Squared}
    """

    def __init__(self, y_pred: np.ndarray, y_true: np.ndarray):
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
        self._y_pred = y_pred.copy()
        self._y_true = y_true.copy()
        self._pearsonr = pearsonr(self._y_pred, self._y_true)
        self._spearmanr = spearmanr(self._y_pred, self._y_true)
        self._mse = mean_squared_error(self._y_pred, self._y_true)
        self._mad = mean_absolute_error(self._y_pred, self._y_true)
        self._rsquared = r2_score(self._y_pred, self._y_true)
        self._dict_indexable_by_key = {
            'mse': self._mse,
            'mad': self._mad,
            'pearsonr': self._pearsonr,
            'spearmanr': self._spearmanr,
            'r2': self._rsquared
        }
        self._dict_indexable_by_int = {
            0: self._mse,
            1: self._mad,
            2: self._pearsonr,
            3: self._spearmanr,
            4: self._rsquared
        }

    def __getitem__(self, index: Literal[0, 1, 2, 3, 4] | \
                    Literal['mse', 'mad', 'pearsonr', 'spearmanr', 'r2']):
        """
        Indexes into RegressionScorer. RegressionScorer is indexable by integer 
        in the following order: 
        (MSE, MAD, Pearson Correlation, Spearman Correlation, R Squared). 
        RegressionScorer is indexable by a string key similar to the following 
        dictionary:
        {'mse': MSE, 'mad': MAD, 'pearsonr': Pearson Correlation, 'spearmanr': 
        Spearman Correlation, 'r2': R Squared}

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
            return self._dict_indexable_by_key[index]
        else:
            raise ValueError(f'Invalid input: {index}.')
        
    
