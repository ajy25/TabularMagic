from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

from ..base_model import BaseDiscriminativeModel, HyperparameterSearcher
from ....data.datahandler import DataEmitter
from ....metrics.classification_scoring import (
    ClassificationMulticlassScorer,
    ClassificationBinaryScorer,
)
from ....feature_selection import BaseFSC, VotingSelectionReport

import numpy as np


class BaseC(BaseDiscriminativeModel):
    """Class that provides the framework that all TabularMagic classification
    classes inherit.

    The primary purpose of BaseC is to automate the scoring and
    model selection processes.
    """

    def __init__(self):
        """Initializes a BaseC object."""
        self._label_encoder = LabelEncoder()
        self._hyperparam_searcher: HyperparameterSearcher = None
        self._estimator: BaseEstimator = None
        self._dataemitter = None
        self._dataemitters = None
        self._feature_selectors = None
        self._max_n_features = 10
        self._name = "BaseC"
        self._train_scorer = None
        self.cv_scorer = None
        self._test_scorer = None
        self._voting_selection_report = None

        # By default, the first column is NOT dropped unless binary. For LinearR,
        # the first column is dropped to avoid multicollinearity.
        self._dropfirst = False

    def specify_data(
        self,
        dataemitter: DataEmitter | None = None,
        dataemitters: list[DataEmitter] | None = None,
        feature_selectors: list[BaseFSC] | None = None,
        max_n_features: int | None = None,
    ):
        """Adds a DataEmitter object to the model.

        Parameters
        ----------
        dataemitter : DataEmitter.
            Default: None.
            DataEmitter that contains the data. If not None, re-specifies the
            DataEmitter for the model.
        dataemitters : list[DataEmitter].
            Default: None.
            If not None, re-specifies the DataEmitters for nested cross validation.
        feature_selectors : list[BaseFSC].
            Default: None.
            If not None, re-specifies the feature selectors
            for the VotingSelectionReport.
        max_n_features : int.
            Default: Mone.
            Maximum number of features to select. Only useful if feature_selectors
            is not None. If not None, re-specifies the maximum number of features.
        """
        if dataemitter is not None:
            self._dataemitter = dataemitter
        if dataemitters is not None:
            self._dataemitters = dataemitters
        if feature_selectors is not None:
            self._feature_selectors = feature_selectors
        if max_n_features is not None:
            self._max_n_features = max_n_features

    def fit(self, verbose: bool = False):
        """Fits the model. Records training metrics, which can be done via
        nested cross validation.

        Parameters
        ----------
        verbose : bool.
            Default: False. If True, prints progress.
        """
        is_binary = True

        if self._dataemitters is None and self._dataemitter is not None:
            X_train_df, y_train_series = self._dataemitter.emit_train_Xy()
            y_train = y_train_series.to_numpy()

            if self._feature_selectors is not None:
                self._voting_selection_report = VotingSelectionReport(
                    selectors=self._feature_selectors,
                    dataemitter=self._dataemitter,
                    max_n_features=self._max_n_features,
                    verbose=verbose,
                )
                X_train = self._voting_selection_report._emit_train_X().to_numpy()
            else:
                X_train = X_train_df.to_numpy()

            y_train_encoded = self._label_encoder.fit_transform(y_train)

            if not np.isin(np.unique(y_train), [0, 1]).all():
                is_binary = False

            self._hyperparam_searcher.fit(X_train, y_train_encoded, verbose)
            self._estimator = self._hyperparam_searcher._best_estimator

            y_pred = self._label_encoder.inverse_transform(
                self._estimator.predict(X_train)
            )

            if hasattr(self._estimator, "predict_proba"):
                y_pred_score = self._estimator.predict_proba(X_train)
            elif hasattr(self._estimator, "decision_function"):
                y_pred_score = self._estimator.decision_function(X_train)

            if not is_binary:
                self._train_scorer = ClassificationMulticlassScorer(
                    y_pred=y_pred,
                    y_true=y_train,
                    y_pred_score=y_pred_score,
                    y_pred_class_order=self._label_encoder.inverse_transform(
                        self._estimator.classes_
                    ),
                    name=str(self),
                )

            else:
                self._train_scorer = ClassificationBinaryScorer(
                    y_pred=y_pred,
                    y_true=y_train,
                    y_pred_score=y_pred_score,
                    name=str(self),
                )

        elif self._dataemitters is not None and self._dataemitter is not None:
            y_preds = []
            y_trues = []
            y_pred_scores = []

            for emitter in self._dataemitters:
                (
                    X_train_df,
                    y_train_series,
                    X_test_df,
                    y_test_series,
                ) = emitter.emit_train_test_Xy()
                y_train = y_train_series.to_numpy()
                y_train_encoded: np.ndarray = self._label_encoder.fit_transform(y_train)

                if self._feature_selectors is not None:
                    fold_selector = VotingSelectionReport(
                        selectors=self._feature_selectors,
                        dataemitter=emitter,
                        max_n_features=self._max_n_features,
                        verbose=verbose,
                    )
                    X_train = fold_selector._emit_train_X().to_numpy()
                    X_test = fold_selector._emit_test_X().to_numpy()
                else:
                    X_train = X_train_df.to_numpy()
                    X_test = X_test_df.to_numpy()

                if not np.isin(np.unique(y_train), [0, 1]).all():
                    is_binary = False

                y_test = y_test_series.to_numpy()

                self._hyperparam_searcher.fit(X_train, y_train_encoded, verbose)
                fold_estimator = self._hyperparam_searcher._best_estimator

                y_pred = self._label_encoder.inverse_transform(
                    fold_estimator.predict(X_test)
                )

                y_preds.append(y_pred)
                y_trues.append(y_test)

                if hasattr(fold_estimator, "predict_proba"):
                    y_pred_scores.append(fold_estimator.predict_proba(X_test))
                elif hasattr(self._estimator, "decision_function"):
                    y_pred_scores.append(self._estimator.decision_function(X_test))

            if len(y_pred_scores) == 0:
                y_pred_scores = None

            if not is_binary:
                self.cv_scorer = ClassificationMulticlassScorer(
                    y_pred=y_preds,
                    y_true=y_trues,
                    y_pred_score=y_pred_scores,
                    y_pred_class_order=self._label_encoder.inverse_transform(
                        fold_estimator.classes_
                    ),
                    name=str(self),
                )
            else:
                self.cv_scorer = ClassificationBinaryScorer(
                    y_pred=y_preds,
                    y_true=y_trues,
                    y_pred_score=y_pred_scores,
                    name=str(self),
                )

            # refit on all data
            X_train_df, y_train_series = self._dataemitter.emit_train_Xy()
            y_train = y_train_series.to_numpy()

            if self._feature_selectors is not None:
                self._voting_selection_report = VotingSelectionReport(
                    selectors=self._feature_selectors,
                    dataemitter=self._dataemitter,
                    max_n_features=self._max_n_features,
                    verbose=verbose,
                )
                X_train = self._voting_selection_report._emit_train_X().to_numpy()
            else:
                X_train = X_train_df.to_numpy()

            y_train_encoded = self._label_encoder.fit_transform(y_train)

            self._hyperparam_searcher.fit(X_train, y_train_encoded, verbose)
            self._estimator = self._hyperparam_searcher._best_estimator

            y_pred = self._label_encoder.inverse_transform(
                self._estimator.predict(X_train)
            )
            if hasattr(self._estimator, "predict_proba"):
                y_pred_score = self._estimator.predict_proba(X_train)
            elif hasattr(self._estimator, "decision_function"):
                y_pred_score = self._estimator.decision_function(X_train)

            if not is_binary:
                self._train_scorer = ClassificationMulticlassScorer(
                    y_pred=y_pred,
                    y_true=y_train,
                    y_pred_score=y_pred_score,
                    y_pred_class_order=self._label_encoder.inverse_transform(
                        self._estimator.classes_
                    ),
                    name=str(self),
                )
            else:
                self._train_scorer = ClassificationBinaryScorer(
                    y_pred=y_pred,
                    y_true=y_train,
                    y_pred_score=y_pred_score,
                    name=str(self),
                )

        else:
            raise ValueError("The datahandler must not be None")

        X_test_df, y_test_series = self._dataemitter.emit_test_Xy()
        y_test = y_test_series.to_numpy()

        if self._feature_selectors is None:
            X_test = X_test_df.to_numpy()
        else:
            X_test = self._voting_selection_report._emit_test_X().to_numpy()

        y_pred = self._label_encoder.inverse_transform(self._estimator.predict(X_test))

        y_pred_score = None
        if hasattr(self._estimator, "predict_proba"):
            y_pred_score = self._estimator.predict_proba(X_test)
        elif hasattr(self._estimator, "decision_function"):
            y_pred_score = self._estimator.decision_function(X_test)

        if not is_binary:
            self._test_scorer = ClassificationMulticlassScorer(
                y_pred=y_pred,
                y_true=y_test,
                y_pred_score=y_pred_score,
                y_pred_class_order=self._label_encoder.inverse_transform(
                    self._estimator.classes_
                ),
                name=str(self),
            )

        else:
            self._test_scorer = ClassificationBinaryScorer(
                y_pred=y_pred, y_true=y_test, y_pred_score=y_pred_score, name=str(self)
            )

    def sklearn_estimator(self):
        """Returns the sklearn estimator object.

        Returns
        -------
        BaseEstimator
        """
        return self._estimator

    def hyperparam_searcher(self) -> HyperparameterSearcher:
        """Returns the HyperparameterSearcher object.

        Returns
        -------
        HyperparameterSearcher
        """
        return self._hyperparam_searcher

    def _set_voting_selection_report(
        self, voting_selection_report: VotingSelectionReport
    ):
        """Adds a VotingSelectionReport object to the model. The VotingSelectionReport
        must have already been fitted to the data.

        Parameters
        ----------
        voting_selection_report : VotingSelectionReport
            The VotingSelectionReport object that has already been fitted to the data.
        """
        self._voting_selection_report = voting_selection_report

    def feature_selection_report(self) -> VotingSelectionReport | None:
        """Returns the VotingSelectionReport object.

        Returns
        -------
        VotingSelectionReport | None
        """
        return self._voting_selection_report

    def _is_cross_validated(self) -> bool:
        """Returns True if the model is cross-validated.

        Returns
        -------
        bool.
            True if the model is cross-validated.
        """
        return self._dataemitters is not None

    def __str__(self):
        return self._name
