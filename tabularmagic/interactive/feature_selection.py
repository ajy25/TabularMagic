import pandas as pd
from ..preprocessing.regression_feature_selection import RegressionBaseSelector
from typing import Iterable

class VotingSelectionReport():
    """
    VotingSelectionReport: generates feature selection-relevant tables.
    """

    def __init__(self, df: pd.DataFrame, X_vars: list[str], y_var: str, 
                 selectors: Iterable[RegressionBaseSelector], 
                 n_target_features: int, verbose: bool = True):
        """
        Initializes a VotingSelectionReport object. 
        VotingSelectionReport selects features via voting selection.

        - df : pd.DataFrame.
        - X_vars : list[str].
            A list of features to look through. 
        - y_var : str.
            The variable to be predicted.
        - selectors : Iterable[BaseSelector].
            Each BaseSelector decides on the top n_target_features. 
        - n_target_features : int. 
            Number of desired features, < len(X_vars).
        - verbose : bool.
            If true, prints progress.
        """
        self._selector_to_support = {}
        for i, selector in enumerate(selectors):
            if verbose:
                print(f'Task {i+1} of {len(selectors)}.\tFitting {selector}.')
            _, support = selector.select(df, X_vars, y_var, n_target_features)
            self._selector_to_support[str(selector)] = support
        self.votes_df = pd.DataFrame.from_dict(self._selector_to_support, 
            orient='index', columns=X_vars)
        self._vote_counts_series = self.votes_df.sum(axis=0)
        
        self._selector_dict_indexable_by_str = {
            str(selector): selector for selector in selectors
        }
        self.top_features = self._vote_counts_series.\
            sort_values(ascending=False).index.to_list()[:n_target_features]


    def __getitem__(self, index: str) -> RegressionBaseSelector:
        """Returns the RegressionBaseSelector by nickname index.
        """
        return self._selector_dict_indexable_by_str[index]



