from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
from typing import Literal
from ...display.print_utils import print_wrapped, quote_and_color
from .base_cluster import BaseClust


class KMeansClust(BaseClust):
    """Class for KMeans clustering."""

    def __init__(
        self,
        k: int | None = None,
        max_k: int = 10,
        criterion: Literal["inertia", "silhouette"] = "silhouette",
        model_random_state: int = 42,
        name: str | None = None,
    ) -> None:
        """Initializes KMeansClust.

        Parameters
        ----------
        k : int | None
            Number of clusters to fit.
            Must be at least 2.
            If None, the number of clusters is selected automatically
            based on the criterion.

        max_k : int
            Maximum number of clusters to fit.
            This parameter is only used when k is None.
            A grid search is performed from 2 to max_k to minimize
            the criterion.
            Default is 10.

        criterion : Literal["inertia", "silhouette"]
            Default: "silhouette".
            The criterion to use when selecting the number of clusters.
            The following are maxmimized:
            - "inertia": negative of the sum of squared distances of samples
                to their closest cluster center.
            - "silhouette": mean silhouette score.

        model_random_state : int
            Random state for the model. Default is 42.

        name : str | None
            Name of the model. Determines how
            the model is named in the report.
            If None, a default name is assigned.
        """
        super().__init__()
        if k is not None and k < 2:
            raise ValueError(
                f"Number of clusters must be at least 2. "
                f"Received: {quote_and_color(k)}."
            )

        self._k = k
        if name is not None:
            self._name = name
        else:
            if k is None:
                self._name = f"KMeansClust(auto, max={max_k}, criterion={criterion})"
            else:
                self._name = f"KMeansClust({k})"
        self._max_k = max_k
        self._criterion = criterion
        self._model_random_state = model_random_state

    def fit(self, verbose: bool = False) -> None:
        """Fits the model to the data."""

        X_train = self._dataemitter.emit_train_X(verbose=verbose)
        X_test = self._dataemitter.emit_test_X(verbose=verbose)

        if isinstance(self._k, int):
            k = self._k

        else:
            # grid search over k to maximize criterion
            if self._criterion == "inertia":
                # we are maximizing the criterion, so we can just compute score
                # which is negative of inertia
                criterion_func = (
                    lambda k: KMeans(
                        n_clusters=k, random_state=self._model_random_state
                    )
                    .fit(X_train)
                    .score(X_train)
                )
            elif self._criterion == "silhouette":
                criterion_func = lambda k: silhouette_score(
                    X_train,
                    KMeans(
                        n_clusters=k, random_state=self._model_random_state
                    ).fit_predict(X_train),
                )
            else:
                raise ValueError(
                    f"Invalid criterion: {quote_and_color(self._criterion)}."
                )

            best_k = 2
            best_metric = (
                float("-inf") if self._criterion == "inertia" else float("-inf")
            )
            for k in range(2, self._max_k + 1):
                metric = criterion_func(k)
                if metric > best_metric:
                    best_k = k
                    best_metric = metric
            k = best_k

        self._estimator = KMeans(
            n_clusters=k,
            random_state=self._model_random_state,
        )

        self._n_clusters = k

        self._train_labels = self._estimator.fit_predict(X_train)
        self._test_labels = self._estimator.predict(X_test)

        if verbose:
            print_wrapped(
                f"Identified {self._n_clusters} optimal clusters for "
                f"{quote_and_color(self._name, 'blue')} "
                f"via {self._criterion}.",
                type="PROGRESS",
            )

        # convert labels to Series
        self._train_labels = pd.Series(
            self._train_labels, index=X_train.index, name="Label"
        )
        self._test_labels = pd.Series(
            self._test_labels, index=X_test.index, name="Label"
        )
