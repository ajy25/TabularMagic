from sklearn.model_selection._search import BaseSearchCV
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import numpy as np
from .base import BaseC
from ....metrics import (
    ClassificationMulticlassScorer,
    ClassificationBinaryScorer,
)
from ....display.print_utils import print_wrapped


class CustomC(BaseC):
    """TabularMagic-compatible interface for user-designed scikit-learn
    estimators/searches/pipelines for classification.

    Hyperparameter search is not conducted unless provided by the
    estimator.
    """

    def __init__(
        self,
        estimator: BaseEstimator | BaseSearchCV | Pipeline,
        name: str | None = None,
    ):
        """Initializes a CustomC object.

        Parameters
        ----------
        estimator : BaseEstimator | BaseSearchCV | Pipeline
            The estimator to be used. Must have a fit method and a
            predict method.

        name : str
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
            X_train = X_train_df
            y_train = y_train_series.to_numpy()

            y_train_encoded = self._label_encoder.fit_transform(y_train)

            if len(self._label_encoder.classes_) > 2:
                self._is_binary = False

            if verbose:
                print_wrapped(f"Fitting {self._name}.", type="PROGRESS")
            self._estimator.fit(X_train, y_train_encoded)

            y_pred_encoded = self._estimator.predict(X_train)

            if hasattr(self._estimator, "predict_proba"):
                y_pred_score = self._estimator.predict_proba(X_train)
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
                X_train = X_train_df
                y_train = y_train_series.to_numpy()

                y_train_encoded: np.ndarray = self._label_encoder.fit_transform(y_train)

                if len(self._label_encoder.classes_) > 2:
                    self._is_binary = False

                X_test = X_test_df
                y_test = y_test_series.to_numpy()

                self._estimator.fit(X_train, y_train_encoded)
                y_pred_encoded = self._estimator.predict(X_test)

                y_preds_encoded.append(y_pred_encoded)
                y_trues.append(y_test)

                if hasattr(self._estimator, "predict_proba"):
                    y_pred_scores.append(self._estimator.predict_proba(X_test))
                elif hasattr(self._estimator, "decision_function"):
                    y_pred_scores.append(self._estimator.decision_function(X_test))

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
                        self._estimator.classes_
                    ),
                    name=str(self),
                )
            # refit on all data
            X_train_df, y_train_series = self._dataemitter.emit_train_Xy()
            X_train = X_train_df
            y_train = y_train_series.to_numpy()

            y_train_encoded = self._label_encoder.fit_transform(y_train)

            self._estimator.fit(X_train, y_train_encoded)

            y_pred_encoded = self._estimator.predict(X_train)
            if hasattr(self._estimator, "predict_proba"):
                y_pred_score = self._estimator.predict_proba(X_train)
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
        X_test = X_test_df
        y_test = y_test_series.to_numpy()

        y_pred_encoded = self._estimator.predict(X_test)

        y_pred_score = None
        if hasattr(self._estimator, "predict_proba"):
            y_pred_score = self._estimator.predict_proba(X_test)
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

    def sklearn_pipeline(self) -> Pipeline:
        """Returns an sklearn pipeline object. The pipelien allows for 
        retrieving model predictions directly from data formatted like the original
        train and test data.
        
        It is not recommended to use TabularMagic for ML production.
        We recommend using TabularMagic to quickly identify promising models
        and then manually implementing and training
        the best model in a production environment.

        Returns
        -------
        Pipeline
        """
        if isinstance(self._estimator, Pipeline):
            new_step = (
                "custom_prep_data", 
                self._dataemitter.sklearn_preprocessing_transformer()
            )
            new_pipeline = Pipeline(
                steps=[new_step, ("model", self._estimator)]
            )
            return new_pipeline
        else:
            pipeline = Pipeline(
                steps=[
                    (
                        "custom_prep_data",
                        self._dataemitter.sklearn_preprocessing_transformer(),
                    ),
                    ("model", self._estimator),
                ]
            )
            return pipeline

    def hyperparam_searcher(self):
        """Raises NotImplementedError. Not implemented for CustomC."""
        raise NotImplementedError("CustomC has no HyperparameterSearcher.")
