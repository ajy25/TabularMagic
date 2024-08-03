from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from typing import Literal

from ..data.datahandler import DataEmitter
from .base_feature_selection import BaseFSC


class KBestFSC(BaseFSC):
    """Selects the k best features based on the f_classif or mutual info
    regression score.
    """

    def __init__(
        self,
        scorer: Literal["f_classif", "mutual_info_classif", "chi2"],
        k: int,
        name: str | None = None,
    ):
        """
        Constructs a KBestFSC.

        Parameters
        ----------
        scorer : Literal['f_classif', 'mutual_info_classif'].
        k : int.
            Number of desired features, < n_predictors.
        name : str.
            Default: None. If None, then outputs the class name.
        """
        if name is None:
            name = f"KBestFSC({scorer})"
        super().__init__(name)
        self._scorer = scorer
        self._k = k

    def select(self, dataemitter: DataEmitter):
        """
        Selects the top max_n_features features
        based on the training data.

        Parameters
        ----------
        dataemitter : DataEmitter.

        Returns
        -------
        np.ndarray ~ (n_in_features).
            All features (variable names).
        np.ndarray ~ (n_out_features).
            Selected features.
        np.ndarray ~ (n_in_features).
            Boolean mask, the support for selected features.
        """
        scorer = None
        if self._scorer == "f_classif":
            scorer = f_classif
        elif self._scorer == "mutual_info_classif":
            scorer = mutual_info_classif
        elif self._scorer == "chi2":
            scorer = chi2
        else:
            raise ValueError(f"Invalid scorer: {self._scorer}")
        selector = SelectKBest(scorer, k=self._k)

        X_train, y_train = dataemitter.emit_train_Xy()
        self._all_features = X_train.columns.to_list()
        selector.fit(X=X_train, y=y_train)

        self._selected_features = selector.get_feature_names_out()
        self._all_feature_scores = selector.scores_
        self._support = selector.get_support()
        self._selected_feature_scores = selector.scores_[self._support]
        return self._all_features, self._selected_features, self._support
