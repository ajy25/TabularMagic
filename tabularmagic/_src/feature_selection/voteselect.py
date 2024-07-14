import pandas as pd
from typing import Iterable
from . import BaseFS
from ..data.datahandler import DataEmitter
from ..display.print_utils import print_wrapped


class VotingSelectionReport:
    """Class for generating feature selection-relevant tables."""

    def __init__(
        self,
        selectors: Iterable[BaseFS],
        dataemitter: DataEmitter,
        max_n_features: int,
        verbose: bool = True,
    ):
        """Initializes a VotingSelectionReport object.
        VotingSelectionReport selects features via voting selection.

        Parameters
        ----------
        selectors : Iterable[BaseSelector].
            Each BaseSelector decides on a maximum of max_n_features.
        dataemitter : DataEmitter.
            The DataEmitter object that contains the data.
        max_n_features : int.
            Number of desired features. 0 < max_n_features < n_predictors.
        verbose : bool.
            Default: True. If True, prints progress.
        """
        self._selector_to_support = {}
        self._emitter = dataemitter

        for selector in selectors:
            if verbose:
                print_wrapped(f"Fitting {selector}.", type="PROGRESS")
            features, _, support = selector.select(self._emitter, max_n_features)
            self._selector_to_support[str(selector)] = support
        self._all_features = features

        self._votes_df = pd.DataFrame.from_dict(
            self._selector_to_support, orient="index", columns=features
        )
        self._vote_counts_series = self._votes_df.sum(axis=0)

        self._selector_dict_indexable_by_str = {
            str(selector): selector for selector in selectors
        }
        self._top_features = self._vote_counts_series.sort_values(
            ascending=False
        ).index.to_list()[:max_n_features]

    def top_features(self) -> list:
        """Returns a list of top features determined by the voting
        selectors."""
        return self._top_features

    def all_features(self) -> list:
        """Returns a list of all features considered by the voting
        selectors."""
        return self._all_features

    def votes_df(self) -> pd.DataFrame:
        """Returns a DataFrame that describes the distribution of
        votes among selectors."""
        return self._votes_df

    def emit_train_X(self) -> pd.DataFrame:
        """Returns the training DataFrame with only the top features."""
        return self._emitter.emit_train_Xy()[0][self._top_features]

    def emit_test_X(self) -> pd.DataFrame:
        """Returns the test DataFrame with only the top features."""
        return self._emitter.emit_test_Xy()[0][self._top_features]

    def __getitem__(self, index: str) -> BaseFS:
        """Returns the RegressionBaseSelector by nickname index."""
        return self._selector_dict_indexable_by_str[index]
