from sklearn.model_selection._search import BaseSearchCV
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import numpy as np
from .base import BaseC
from ....metrics.classification_scoring import (
    ClassificationMulticlassScorer,
    ClassificationBinaryScorer,
)


class CustomC(BaseC):
    """TabularMagic-compatible wrapper for user-designed scikit-learn
    estimators/searches/pipelines.

    Hyperparameter search is not conducted unless provided by the
    estimator.
    """

    def __init__(
        self, estimator: BaseEstimator | BaseSearchCV | Pipeline, name: str = None
    ):
        """Initializes a CustomC object.

        Parameters
        ----------
        estimator : BaseEstimator | BaseSearchCV | Pipeline.
            The estimator to be used. Must have a fit method and a
            predict method.
        name : str.
            Default: None.
            The name of the model. If None, the estimator's
            __str__() implementation is used.
        """
        super().__init__()
        self._estimator: BaseSearchCV | BaseEstimator | Pipeline = estimator
        if name is None:
            self._name = str(estimator)
        else:
            self._name = name

    def fit(self):
        """Fits the model. Records training metrics, which can be done via
        nested cross validation.
        """
        is_binary = False

        if self._dataemitters is None and self._dataemitter is not None:
            X_train_df, y_train_series = self._dataemitter.emit_train_Xy()
            X_train = X_train_df.to_numpy()
            y_train = y_train_series.to_numpy()

            y_train_encoded = self._label_encoder.fit_transform(y_train)

            if np.isin(np.unique(y_train), [0, 1]).all():
                is_binary = True

            self._estimator.fit(X_train, y_train_encoded)

            y_pred = self._label_encoder.inverse_transform(
                self._estimator.predict(X_train)
            )

            if hasattr(self._estimator, "predict_proba"):
                y_pred_score = self._estimator.predict_proba(X_train)
            elif hasattr(self._estimator, "decision_function"):
                y_pred_score = self._estimator.decision_function(X_train)

            if not is_binary:
                self.train_scorer = ClassificationMulticlassScorer(
                    y_pred=y_pred,
                    y_true=y_train,
                    y_pred_score=y_pred_score,
                    y_pred_class_order=self._label_encoder.inverse_transform(
                        self._estimator.classes_
                    ),
                    name=str(self),
                )

            else:
                self.train_scorer = ClassificationBinaryScorer(
                    y_pred=y_pred,
                    y_true=y_train,
                    y_pred_score=y_pred_score,
                    name=str(self),
                )

        elif self._dataemitters is not None and self._dataemitter is not None:
            y_preds = []
            y_trues = []
            y_pred_scores = []

            is_binary = True

            for emitter in self._dataemitters:
                (
                    X_train_df,
                    y_train_series,
                    X_test_df,
                    y_test_series,
                ) = emitter.emit_train_test_Xy()
                X_train = X_train_df.to_numpy()
                y_train = y_train_series.to_numpy()

                y_train_encoded: np.ndarray = self._label_encoder.fit_transform(y_train)

                if not np.isin(np.unique(y_train), [0, 1]).all():
                    is_binary = False

                X_test = X_test_df.to_numpy()
                y_test = y_test_series.to_numpy()

                self._estimator.fit(X_train, y_train_encoded)
                y_pred = self._label_encoder.inverse_transform(
                    self._estimator.predict(X_test)
                )

                y_preds.append(y_pred)
                y_trues.append(y_test)

                if hasattr(self._estimator, "predict_proba"):
                    y_pred_scores.append(self._estimator.predict_proba(X_test))
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
                        self._estimator.classes_
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
            X_train = X_train_df.to_numpy()
            y_train = y_train_series.to_numpy()

            y_train_encoded = self._label_encoder.fit_transform(y_train)

            self._estimator.fit(X_train, y_train_encoded)

            y_pred = self._label_encoder.inverse_transform(
                self._estimator.predict(X_train)
            )
            if hasattr(self._estimator, "predict_proba"):
                y_pred_score = self._estimator.predict_proba(X_train)
            elif hasattr(self._estimator, "decision_function"):
                y_pred_score = self._estimator.decision_function(X_train)

            if not is_binary:
                self.train_scorer = ClassificationMulticlassScorer(
                    y_pred=y_pred,
                    y_true=y_train,
                    y_pred_score=y_pred_score,
                    y_pred_class_order=self._label_encoder.inverse_transform(
                        self._estimator.classes_
                    ),
                    name=str(self),
                )
            else:
                self.train_scorer = ClassificationBinaryScorer(
                    y_pred=y_pred,
                    y_true=y_train,
                    y_pred_score=y_pred_score,
                    name=str(self),
                )

        else:
            raise ValueError("The datahandler must not be None")

        X_test_df, y_test_series = self._dataemitter.emit_test_Xy()
        X_test = X_test_df.to_numpy()
        y_test = y_test_series.to_numpy()

        y_pred = self._label_encoder.inverse_transform(self._estimator.predict(X_test))

        y_pred_score = None
        if hasattr(self._estimator, "predict_proba"):
            y_pred_score = self._estimator.predict_proba(X_test)
        elif hasattr(self._estimator, "decision_function"):
            y_pred_score = self._estimator.decision_function(X_test)

        if not is_binary:
            self.test_scorer = ClassificationMulticlassScorer(
                y_pred=y_pred,
                y_true=y_test,
                y_pred_score=y_pred_score,
                y_pred_class_order=self._label_encoder.inverse_transform(
                    self._estimator.classes_
                ),
                name=str(self),
            )

        else:
            self.test_scorer = ClassificationBinaryScorer(
                y_pred=y_pred, y_true=y_test, y_pred_score=y_pred_score, name=str(self)
            )

    def hyperparam_searcher(self):
        """Raises NotImplementedError. Not implemented for CustomC."""
        raise NotImplementedError("CustomR has no HyperparameterSearcher.")
