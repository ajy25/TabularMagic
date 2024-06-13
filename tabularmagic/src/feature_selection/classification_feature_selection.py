from sklearn.feature_selection import (SelectKBest, 
    f_classif, mutual_info_classif)
import pandas as pd
from typing import Literal

from ..data.datahandler import DataEmitter



class BaseFeatureSelectorR():
    """A feature selection class.
    """

    def __init__(self, name: str = None):
        """
        Constructs a BaseFeatureSelectorR.

        Parameters
        ----------
        - nickname

        Returns
        -------
        - None
        """
        self._name = name


    def select(self, 
               dataemitter: DataEmitter,
               n_target_features: int):
        """
        Selects the top n_target_features features 
        based on the training data.

        Parameters
        ----------
        - dataemitter : DataEmitter.
        - n_target_features : int. 
            Number of desired features, < n_predictors.

        Returns
        -------
        - np array ~ (n_features).
            Selected features.
        - np array ~ (n_features).
            Boolean mask.
        """
        return None, None


    def __str__(self):
        return self._name





class KBestSelectorR(BaseFeatureSelectorR):
    """Selects the k best features based on the f_regression or mutual info 
    regression score.
    """

    def __init__(self, 
                 scorer: Literal['f_classif', 'mutual_info_classif'], 
                 name: str = None):
        """
        Constructs a KBestSelectorR.

        Parameters
        ----------
        - scorer : Literal['f_classif', 'mutual_info_classif'].
        - nickname : str.
            Default: None. If None, then outputs the class name. 

        Returns
        -------
        - None
        """
        super().__init__(name)
        if self._name is None:
            self._name = f'KBestSelector({scorer})'
        self.scorer = scorer


    def select(self, 
               dataemitter: DataEmitter,
               n_target_features: int):
        """
        Selects the top n_target_features features
        based on the training data.

        Parameters
        ----------
        - dataemitter : DataEmitter.
        - n_target_features : int. 
            Number of desired features, < n_predictors.

        Returns
        -------
        - np array ~ (n_features).
            Selected features.
        - np array ~ (n_features).
            Boolean mask.
        """
        scorer = None
        if self.scorer == 'f_classification':
            scorer = f_classif
        elif self.scorer == 'mutual_info_classification':
            scorer = mutual_info_classif
        selector = SelectKBest(scorer, k=n_target_features)

        X_train, y_train = dataemitter.emit_train_Xy()

        selector.fit(
            X=X_train,
            y=y_train
        )

        self.selected_features = selector.get_feature_names_out()
        self.all_feature_scores = selector.scores_
        self.support = selector.get_support()
        self.selected_feature_scores = selector.scores_[self.support]
        return self.selected_features, self.support





