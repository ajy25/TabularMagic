from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, f1_score

from ..base_model import BaseDiscriminativeModel, HyperparameterSearcher
from ....data.datahandler import DataEmitter
from ....metrics import (
    ClassificationMulticlassScorer,
    ClassificationBinaryScorer,
)
from ....feature_selection import BaseFSC, VotingSelectionReport
from ....display.print_utils import print_wrapped
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
        self._max_n_features = None
        self._name = "BaseC"
        self._train_scorer = None
        self._cv_scorer = None
        self._test_scorer = None
        self._feature_selection_report = None
        self._predictors = None
        self._is_binary = True
        self._threshold = None

        # By default, the first level is NOT dropped unless binary. For linear models,
        # the first level is dropped to avoid multicollinearity.
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
            Default: None.
            Maximum number of features to select.
            Only useful if feature_selectors is not None.
            If None, then all features with at least 50% support are selected.
        """
        if dataemitter is not None:
            self._dataemitter = dataemitter
        if dataemitters is not None:
            self._dataemitters = dataemitters
        if feature_selectors is not None:
            self._feature_selectors = feature_selectors
        if max_n_features != "ignore":
            self._max_n_features = max_n_features

    def fit(self, verbose: bool = False):
        """Fits the model. Records training metrics, which can be done via
        nested cross validation.

        Parameters
        ----------
        verbose : bool.
            Default: False. If True, prints progress.
        """
        self._is_binary = True

        if self._dataemitters is None and self._dataemitter is not None:
            X_train_df, y_train_series = self._dataemitter.emit_train_Xy()
            y_train = y_train_series.to_numpy()

            if self._feature_selectors is not None:
                self._feature_selection_report = VotingSelectionReport(
                    selectors=self._feature_selectors,
                    dataemitter=self._dataemitter,
                    max_n_features=self._max_n_features,
                    verbose=verbose,
                )
                self._predictors = self._feature_selection_report.top_features()
                X_train = self._feature_selection_report._emit_train_X().to_numpy()
            else:
                self._predictors = X_train_df.columns.to_list()
                X_train = X_train_df.to_numpy()

            y_train_encoded = self._label_encoder.fit_transform(y_train)

            if len(self._label_encoder.classes_) > 2:
                self._is_binary = False

            self._hyperparam_searcher.fit(X_train, y_train_encoded, verbose)
            self._estimator = self._hyperparam_searcher._best_estimator

            y_pred_encoded = self._estimator.predict(X_train)

            if hasattr(self._estimator, "predict_proba"):
                y_pred_score = self._estimator.predict_proba(X_train)

                if self._is_binary:
                    self._threshold = self._select_optimal_threshold_f1(
                        y_train_encoded, y_pred_score[:, 1]
                    )
                    y_pred_encoded = y_pred_score[:, 1] > self._threshold

                    print_wrapped(
                        f"Optimal threshold set for {self._name} via F1 score.", 
                        type="PROGRESS",
                    )

            elif hasattr(self._estimator, "decision_function"):
                y_pred_score = self._estimator.decision_function(X_train)

            if self._is_binary:

                self._train_scorer = ClassificationBinaryScorer(
                    y_pred=y_pred_encoded,
                    y_true=y_train_encoded,
                    pos_label=self._label_encoder.classes_[1],
                    y_pred_score=y_pred_score,
                    name=str(self),
                )

            else:
                self._train_scorer = ClassificationMulticlassScorer(
                    y_pred=self._label_encoder.inverse_transform(y_pred_encoded),
                    y_true=y_train,
                    y_pred_score=y_pred_score,
                    y_pred_class_order=self._label_encoder.inverse_transform(
                        self._estimator.classes_
                    ),
                    name=str(self),
                )

        elif self._dataemitters is not None and self._dataemitter is not None:
            y_preds_encoded = []
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

                if len(self._label_encoder.classes_) > 2:
                    self._is_binary = False

                y_test = y_test_series.to_numpy()

                self._hyperparam_searcher.fit(X_train, y_train_encoded, verbose)
                fold_estimator = self._hyperparam_searcher._best_estimator

                y_trues.append(y_test)

                fold_threshold = None

                if hasattr(fold_estimator, "predict_proba"):
                    y_pred_score_fold = fold_estimator.predict_proba(X_test)
                    if self._is_binary:
                        fold_threshold = self._select_optimal_threshold_f1(
                            self._label_encoder.transform(y_test), 
                            y_pred_score_fold[:, 1]
                        )
                        y_pred_scores.append(y_pred_score_fold)
                elif hasattr(self._estimator, "decision_function"):
                    y_pred_score_fold = fold_estimator.decision_function(X_test)
                    y_pred_scores.append(y_pred_score_fold)

                if self._is_binary and fold_threshold is not None:
                    y_pred_encoded = y_pred_score_fold[:, 1] > fold_threshold
                else:
                    y_pred_encoded = fold_estimator.predict(X_test)
                y_preds_encoded.append(y_pred_encoded)

                

            if len(y_pred_scores) == 0:
                y_pred_scores = None

            if self._is_binary:
                self._cv_scorer = ClassificationBinaryScorer(
                    y_pred=y_preds_encoded,
                    y_true=[self._label_encoder.transform(y) for y in y_trues],
                    pos_label=self._label_encoder.classes_[1],
                    y_pred_score=y_pred_scores,
                    name=str(self),
                )
            else:
                self._cv_scorer = ClassificationMulticlassScorer(
                    y_pred=[
                        self._label_encoder.inverse_transform(y)
                        for y in y_preds_encoded
                    ],
                    y_true=y_trues,
                    y_pred_score=y_pred_scores,
                    y_pred_class_order=self._label_encoder.inverse_transform(
                        fold_estimator.classes_
                    ),
                    name=str(self),
                )

            # refit on all data
            X_train_df, y_train_series = self._dataemitter.emit_train_Xy()
            y_train = y_train_series.to_numpy()

            if self._feature_selectors is not None:
                self._feature_selection_report = VotingSelectionReport(
                    selectors=self._feature_selectors,
                    dataemitter=self._dataemitter,
                    max_n_features=self._max_n_features,
                    verbose=verbose,
                )
                X_train = self._feature_selection_report._emit_train_X().to_numpy()
                self._predictors = self._feature_selection_report.top_features()
            else:
                self._predictors = X_train_df.columns.to_list()
                X_train = X_train_df.to_numpy()

            y_train_encoded = self._label_encoder.fit_transform(y_train)

            self._hyperparam_searcher.fit(X_train, y_train_encoded, verbose)
            self._estimator = self._hyperparam_searcher._best_estimator

            y_pred_encoded = self._estimator.predict(X_train)
            if hasattr(self._estimator, "predict_proba"):
                y_pred_score = self._estimator.predict_proba(X_train)

                if self._is_binary:
                    self._threshold = self._select_optimal_threshold_f1(
                        y_train_encoded, y_pred_score[:, 1]
                    )
                    y_pred_encoded = y_pred_score[:, 1] > self._threshold

                    print_wrapped(
                        f"Optimal threshold set for {self._name} via F1 score.", 
                        type="PROGRESS",
                    )

            elif hasattr(self._estimator, "decision_function"):
                y_pred_score = self._estimator.decision_function(X_train)

            if self._is_binary:
                self._train_scorer = ClassificationBinaryScorer(
                    y_pred=y_pred_encoded,
                    y_true=y_train_encoded,
                    pos_label=self._label_encoder.classes_[1],
                    y_pred_score=y_pred_score,
                    name=str(self),
                )
            else:
                self._train_scorer = ClassificationMulticlassScorer(
                    y_pred=self._label_encoder.inverse_transform(y_pred_encoded),
                    y_true=y_train,
                    y_pred_score=y_pred_score,
                    y_pred_class_order=self._label_encoder.inverse_transform(
                        self._estimator.classes_
                    ),
                    name=str(self),
                )

        else:
            raise ValueError("The datahandler must not be None")

        X_test_df, y_test_series = self._dataemitter.emit_test_Xy()

        y_test = y_test_series.to_numpy()

        if self._feature_selectors is None:
            X_test = X_test_df.to_numpy()
        else:
            X_test = self._feature_selection_report._emit_test_X().to_numpy()

        y_pred_encoded = self._estimator.predict(X_test)

        y_pred_score = None
        if hasattr(self._estimator, "predict_proba"):
            y_pred_score = self._estimator.predict_proba(X_test)

            if self._is_binary and self._threshold is not None:
                y_pred_encoded = y_pred_score[:, 1] > self._threshold

        elif hasattr(self._estimator, "decision_function"):
            y_pred_score = self._estimator.decision_function(X_test)

        if self._is_binary:
            self._test_scorer = ClassificationBinaryScorer(
                y_pred=y_pred_encoded,
                y_true=self._label_encoder.transform(y_test),
                pos_label=self._label_encoder.classes_[1],
                y_pred_score=y_pred_score,
                name=str(self),
            )
        else:
            self._test_scorer = ClassificationMulticlassScorer(
                y_pred=self._label_encoder.inverse_transform(y_pred_encoded),
                y_true=y_test,
                y_pred_score=y_pred_score,
                y_pred_class_order=self._label_encoder.inverse_transform(
                    self._estimator.classes_
                ),
                name=str(self),
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
        self._feature_selection_report = voting_selection_report

    def feature_selection_report(self) -> VotingSelectionReport | None:
        """Returns the VotingSelectionReport object.

        Returns
        -------
        VotingSelectionReport | None
        """
        if self._feature_selection_report is None:
            print_wrapped(
                f"No feature selection report available for {self._name}.",
                type="WARNING",
            )
        return self._feature_selection_report

    def is_binary(self) -> bool:
        """Returns True if the model is binary.

        Returns
        -------
        bool.
            True if the model is binary.
        """
        return self._is_binary

    def is_cross_validated(self) -> bool:
        """Returns True if the model is cross-validated.

        Returns
        -------
        bool.
            True if the model is cross-validated.
        """
        return self._dataemitters is not None

    def pos_label(self) -> str | None:
        """Returns the positive label.

        Returns
        -------
        str.
            The positive label.
        """
        if self._is_binary:
            return self._label_encoder.classes_[1]
        else:
            print_wrapped(
                "This is not a binary model. No positive label available.",
                type="WARNING",
            )
            return None

    def predictors(self) -> list[str] | None:
        """Returns the predictors.

        Returns
        -------
        list[str].
            The predictors.
        """
        if self._predictors is None:
            print_wrapped(
                "No predictors available. The model has not been fitted.",
                type="WARNING",
            )
        return self._predictors
    
    def _select_optimal_threshold_roc(
        self, 
        fpr: np.ndarray, 
        tpr: np.ndarray, 
        thresholds: np.ndarray
    ) -> float:
        """Selects the optimal threshold for binary classification models.
        The optimal threshold is selected based on the training data via the ROC curve.

        Parameters
        ----------
        fpr : np.ndarray ~ (sample_size,)
            False positive rates.
        tpr : np.ndarray ~ (sample_size,)
            True positive rates.
        thresholds : np.ndarray ~ (sample_size,)
            Thresholds.

        Returns
        -------
        float
            The optimal threshold.
        """
        if not self._is_binary:
            raise ValueError("This method is only for binary classification models.")
        optimal_idx = np.argmax(tpr - fpr)
        return thresholds[optimal_idx]
    

    def _select_optimal_threshold_f1(
        self, 
        y_true: np.ndarray,
        y_pred_score: np.ndarray
    ) -> float:
        """Selects the optimal threshold for binary classification models via 
        the F1 score.
        
        Parameters
        ----------
        y_true : np.ndarray ~ (sample_size,)
            True labels.
        y_pred_score : np.ndarray ~ (sample_size,)
            Predicted scores.

        Returns
        -------
        float
            The optimal threshold.
        """
        if not self._is_binary:
            raise ValueError("This method is only for binary classification models.")
        thresholds = np.linspace(0.1, 0.9, 100)
        f1_scores = []
        for threshold in thresholds:
            y_pred = y_pred_score > threshold
            f1_scores.append(f1_score(y_true, y_pred))
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx]
        


    def __str__(self):
        return self._name
