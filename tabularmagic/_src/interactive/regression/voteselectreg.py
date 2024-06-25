import pandas as pd
from typing import Iterable

from ...feature_selection.regression_feature_selection \
    import BaseFeatureSelectorR
from ...data.datahandler import DataHandler
from ...util.console import print_wrapped


class RegressionVotingSelectionReport():
    """
    RegressionVotingSelectionReport: 
    generates feature selection-relevant tables.
    """

    def __init__(self, 
                 selectors: Iterable[BaseFeatureSelectorR], 
                 datahandler: DataHandler,
                 y_var: str,
                 X_vars: list[str],
                 n_target_features: int, 
                 verbose: bool = True):
        """
        Initializes a RegressionVotingSelectionReport object. 
        RegressionVotingSelectionReport selects features via voting selection.

        Parameters
        ----------
        selectors : Iterable[BaseSelector].
            Each BaseSelector decides on the top n_target_features. 
        datahandler : DataHandler.
            The DataHandler object that contains the data.
        y_var : str.
            The name of the dependent variable.
        X_vars : Iterable[str].
            The names of the independent variables.
        n_target_features : int. 
            Number of desired features, < n_predictors.
        verbose : bool.
            Default: True. If True, prints progress.
        """
        self._selector_to_support = {}
        self._emitter = datahandler.train_test_emitter(
            y_var=y_var, X_vars=X_vars
        )

        for i, selector in enumerate(selectors):
            if verbose:
                print_wrapped(
                    f'Task {i+1} of {len(selectors)}.\tFitting {selector}.', 
                    type='UPDATE')
            _, support = selector.select(self._emitter, n_target_features)
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


    def __getitem__(self, index: str) -> BaseFeatureSelectorR:
        """Returns the RegressionBaseSelector by nickname index.
        """
        return self._selector_dict_indexable_by_str[index]



