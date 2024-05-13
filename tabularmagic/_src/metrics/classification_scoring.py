import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)

class ClassificationScorer:
    """ClassificationScorer: Class built for simple scoring of 
    classification models.
    Only inputs are predicted and true values.
    Capable of scoring cross-validation outputs.

    ClassificationScorer is indexable by integers in the following order:
        (Accuracy, F1 Score, Precision, Recall, ROC AUC).

    ClassificationScorer also is indexable by a string key similar to the
    dictionary: {'accuracy': Accuracy, 'f1': F1 Score, 'precision': Precision,
                 'recall': Recall, 'roc_auc': ROC AUC}.
    """

    def __init__(self, y_pred: np.ndarray | list, 
                 y_true: np.ndarray | list, 
                 y_pred_scores: np.ndarray | list = None, 
                 name: str = None):
        """
        Initializes a ClassificationScorer object.

        Parameters
        ----------
        - y_pred: np.ndarray ~ (sample_size) | list[np.ndarray ~ (sample_size)].
        - y_true: np.ndarray ~ (sample_size) | list[np.ndarray ~ (sample_size)].
        - y_pred_scores: np.ndarray ~ (sample_size, n_classes) |
            list[np.ndarray ~ (sample_size, n_classes)].
        - name: str.

        Returns
        -------
        - None
        """
        if name is None:
            self._name = 'Model'
        else:
            self._name = name
        self._y_pred = y_pred
        self._y_true = y_true
        self._y_pred_scores = y_pred_scores

        self._stats_df = None
        self._cv_stats_df = None
        self._set_stats_df()



    def _set_stats_df(self):
        """
        Creates statistics DataFrames given y_pred and y_true. If y_pred and
        y_true are lists, then the elements are treated as 
        cross-validation folds, and the statistics are averaged 
        across all folds.
        """

        y_pred = self._y_pred
        y_true = self._y_true
        y_pred_scores = self._y_pred_scores

        df = pd.DataFrame(columns=['Statistic', self._name])
        cvdf = pd.DataFrame(columns=['Fold', 'Statistic', self._name])

        if isinstance(y_pred, np.ndarray) and isinstance(y_true, np.ndarray):
            df.loc[len(df)] = pd.Series(
                {
                    'Statistic': 'accuracy',
                    self._name: accuracy_score(y_true, y_pred)
                }
            )
            df.loc[len(df)] = pd.Series(
                {
                    'Statistic': 'f1',
                    self._name: f1_score(y_true, y_pred, average='macro')
                }
            )
            df.loc[len(df)] = pd.Series(
                {
                    'Statistic': 'precision',
                    self._name: precision_score(y_true, y_pred, average='macro', 
                                                zero_division=np.nan)
                }
            )
            df.loc[len(df)] = pd.Series(
                {
                    'Statistic': 'recall',
                    self._name: recall_score(y_true, y_pred, average='macro')
                }
            )
            if y_pred_scores is not None:
                df.loc[len(df)] = pd.Series(
                    {
                        'Statistic': 'roc_auc',
                        self._name: roc_auc_score(y_true, y_pred_scores, 
                            average='macro', multi_class='ovo')
                    }
                )
            df.loc[len(df)] = pd.Series(
                {
                    'Statistic': 'n',
                    self._name: len(y_pred)
                }
            )
            self._stats_df = df.set_index('Statistic')


        elif isinstance(y_pred, list) and isinstance(y_true, list):
            assert len(y_pred) == len(y_true)
            for i, (y_pred_elem, y_true_elem) in enumerate(zip(y_pred, y_true)):
                cvdf.loc[len(cvdf)] = pd.Series(
                    {
                        'Statistic': 'accuracy',
                        self._name: accuracy_score(y_true_elem, y_pred_elem),
                        'Fold': i
                    }
                )
                cvdf.loc[len(cvdf)] = pd.Series(
                    {
                        'Statistic': 'f1',
                        self._name: f1_score(y_true_elem, 
                                             y_pred_elem, average='macro'),
                        'Fold': i
                    }
                )
                cvdf.loc[len(cvdf)] = pd.Series(
                    {
                        'Statistic': 'precision',
                        self._name: precision_score(y_true_elem, 
                            y_pred_elem, average='macro', zero_division=np.nan),
                        'Fold': i
                    }
                )
                cvdf.loc[len(cvdf)] = pd.Series(
                    {
                        'Statistic': 'recall',
                        self._name: recall_score(y_true_elem, 
                            y_pred_elem, average='macro'),
                        'Fold': i
                    }
                )
                if y_pred_scores is not None:
                    cvdf.loc[len(cvdf)] = pd.Series(
                        {
                            'Statistic': 'roc_auc',
                            self._name: roc_auc_score(y_true_elem, 
                                                      y_pred_scores[i], 
                                average='macro', multi_class='ovo'),
                            'Fold': i
                        }
                    )
                cvdf.loc[len(cvdf)] = pd.Series(
                    {
                        'Statistic': 'n',
                        self._name: len(y_pred_elem),
                        'Fold': i
                    }
                )
            
            self._cv_stats_df = cvdf.set_index(['Statistic', 'Fold'])
            self._stats_df = pd.DataFrame(columns=[self._name])
            for stat in cvdf['Statistic'].unique():
                self._stats_df.loc[stat, self._name] = cvdf.loc[
                    cvdf['Statistic'] == stat, self._name].mean()

        else:
            raise ValueError('Input types for y_pred and y_true are invalid.')
        


    def stats_df(self):
        """Outputs a DataFrame that contains the model's evaluation metrics.

        Returns
        -------
        - pd.DataFrame.
        """
        return self._stats_df

    def cv_df(self):
        """Outputs a DataFrame that contains the cross-validation 
        evaluation metrics.

        Returns
        -------
        - pd.DataFrame.
        """
        return self._cv_stats_df