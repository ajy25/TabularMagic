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
        max_n_features: int | None = None,
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
            Default: None.
            Number of desired features. 0 < max_n_features < n_predictors.
            If None, then all features with at least 50% support are selected.
        verbose : bool.
            Default: True. If True, prints progress.
        """
        self._selector_to_support = {}
        self._emitter = dataemitter

        for selector in selectors:
            if verbose:
                print_wrapped(f"Fitting {selector}.", type="PROGRESS")
            features, _, support = selector.select(self._emitter)
            self._selector_to_support[str(selector)] = support
        self._all_features = features

        self._votes_df = pd.DataFrame.from_dict(
            self._selector_to_support, orient="index", columns=features
        )
        self._vote_counts_series = self._votes_df.sum(axis=0)

        self._selector_dict_indexable_by_str = {
            str(selector): selector for selector in selectors
        }
        if max_n_features is not None:
            self._top_features = self._vote_counts_series.sort_values(
                ascending=False
            ).index.to_list()[:max_n_features]
        else:
            self._top_features = self._vote_counts_series[
                self._vote_counts_series >= len(selectors) / 2
            ].index.to_list()

    def top_features(self) -> list:
        """Returns a list of top features determined by the voting
        selectors."""
        return self._top_features

    def all_features(self) -> list:
        """Returns a list of all features considered by the voting
        selectors."""
        return self._all_features

    def votes(self) -> pd.DataFrame:
        """Returns a DataFrame that describes the distribution of
        votes among selectors."""
        return self._votes_df.T

    def _emit_train_X(self) -> pd.DataFrame:
        """Returns the training DataFrame with only the top features."""
        return self._emitter.emit_train_Xy()[0][self._top_features]

    def _emit_test_X(self) -> pd.DataFrame:
        """Returns the test DataFrame with only the top features."""
        return self._emitter.emit_test_Xy()[0][self._top_features]

    def __getitem__(self, index: str) -> BaseFS:
        """Returns the RegressionBaseSelector by nickname index."""
        return self._selector_dict_indexable_by_str[index]
