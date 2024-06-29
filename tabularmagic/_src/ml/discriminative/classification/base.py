from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

from ..base_model import BaseDiscriminativeModel, HyperparameterSearcher
from ....data.datahandler import DataEmitter
from ....metrics.classification_scoring import (
    ClassificationMulticlassScorer,
    ClassificationBinaryScorer,
)

import numpy as np


class BaseC(BaseDiscriminativeModel):
    """Class that provides the framework that all TabularMagic classification
    classes inherit.

    The primary purpose of BaseC is to automate the scoring and
    model selection processes.
    """

    def __init__(self):
        """Initializes a BaseC object. Creates copies of the inputs."""
        self._label_encoder = LabelEncoder()
        self._hyperparam_searcher: HyperparameterSearcher = None
        self._estimator: BaseEstimator = None
        self._dataemitter = None
        self._dataemitters = None
        self._name = "BaseC"
        self.train_scorer = None
        self.cv_scorer = None
        self.test_scorer = None

        # By default, the first column is NOT dropped unless binary. For LinearR,
        # the first column is dropped to avoid multicollinearity.
        self._dropfirst = False

    def specify_data(
        self, dataemitter: DataEmitter, dataemitters: list[DataEmitter] = None
    ):
        """Adds a DataEmitter object to the model.

        Parameters
        ----------
        dataemitter : DataEmitter.
            DataEmitter that contains the data.
        dataemitters : list[DataEmitter].
            Default: None.
            If not None, specifies the DataEmitters for nested cross validation.
        """
        self._dataemitter = dataemitter
        self._dataemitters = dataemitters

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

            self._hyperparam_searcher.fit(X_train, y_train_encoded)
            self._estimator = self._hyperparam_searcher._best_estimator

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

                self._hyperparam_searcher.fit(X_train, y_train_encoded)
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
            X_train = X_train_df.to_numpy()
            y_train = y_train_series.to_numpy()

            y_train_encoded = self._label_encoder.fit_transform(y_train)

            self._hyperparam_searcher.fit(X_train, y_train_encoded)
            self._estimator = self._hyperparam_searcher._best_estimator

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

    def sklearn_estimator(self):
        """Returns the sklearn estimator object.

        Returns
        -------
        - BaseEstimator
        """
        return self._estimator

    def hyperparam_searcher(self) -> HyperparameterSearcher:
        """Returns the HyperparameterSearcher object.

        Returns
        -------
        - HyperparameterSearcher
        """
        return self._hyperparam_searcher

    def _is_cross_validated(self) -> bool:
        """Returns True if the model is cross-validated.

        Returns
        -------
        - bool
        """
        return self._dataemitters is not None

    def __str__(self):
        return self._name
