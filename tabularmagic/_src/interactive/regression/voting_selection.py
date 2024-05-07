import pandas as pd
from ...feature_selection.regression_feature_selection \
    import RegressionBaseSelector
from typing import Iterable
from ...util.console import print_wrapped


class RegressionVotingSelectionReport():
    """
    RegressionVotingSelectionReport: 
    generates feature selection-relevant tables.
    """

    def __init__(self, df: pd.DataFrame, X_vars: list[str], y_var: str, 
                 selectors: Iterable[RegressionBaseSelector], 
                 n_target_features: int, verbose: bool = True):
        """
        Initializes a RegressionVotingSelectionReport object. 
        RegressionVotingSelectionReport selects features via voting selection.

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
                print_wrapped(
                    f'Task {i+1} of {len(selectors)}.\tFitting {selector}.', 
                    type='UPDATE')
            _, support = selector.select(df, X_vars, y_var, n_target_features)
            self._selector_to_support[str(selector)] = support
        self._votes_df = pd.DataFrame.from_dict(self._selector_to_support, 
            orient='index', columns=X_vars)
        self._vote_counts_series = self._votes_df.sum(axis=0)
        
        self._selector_dict_indexable_by_str = {
            str(selector): selector for selector in selectors
        }
        self._top_features = self._vote_counts_series.\
            sort_values(ascending=False).index.to_list()[:n_target_features]


    def top_features(self) -> list:
        """Returns a list of top features determined by the voting 
        selectors."""
        return self._top_features
    
    def votes_df(self) -> pd.DataFrame:
        """Returns a DataFrame that describes the distribution of 
        votes among selectors."""
        return self._votes_df


    def __getitem__(self, index: str) -> RegressionBaseSelector:
        """Returns the RegressionBaseSelector by nickname index.
        """
        return self._selector_dict_indexable_by_str[index]



