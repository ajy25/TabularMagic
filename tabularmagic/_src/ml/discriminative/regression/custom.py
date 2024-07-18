from sklearn.model_selection._search import BaseSearchCV
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from .base import BaseR
from ....metrics import RegressionScorer
from ....display.print_utils import print_wrapped


class CustomR(BaseR):
    """TabularMagic-compatible wrapper for user-designed scikit-learn
    estimators/searches/pipelines.

    Hyperparameter search is not conducted unless provided by the
    estimator.
    """

    def __init__(
        self,
        estimator: BaseEstimator | BaseSearchCV | Pipeline,
        name: str | None = None,
    ):
        """Initializes a CustomR object.

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
        self._estimator = estimator
        if name is None:
            self._name = str(estimator)
        else:
            self._name = name

    def fit(self, verbose: bool = False):
        """Fits the model. Records training metrics, which can be done via
        nested cross validation.
        """
        y_scaler = self._dataemitter.y_scaler()

        if self._dataemitters is None and self._dataemitter is not None:
            X_train_df, y_train_series = self._dataemitter.emit_train_Xy()
            X_train = X_train_df.to_numpy()
            y_train = y_train_series.to_numpy()
            if verbose:
                print_wrapped(f"Fitting {self._name}.", type="PROGRESS")
            self._estimator.fit(X_train, y_train)
            y_pred = self._estimator.predict(X_train)
            if y_scaler is not None:
                y_pred = y_scaler.inverse_transform(y_pred)
                y_train = y_scaler.inverse_transform(y_train)
            self.train_scorer = RegressionScorer(
                y_pred=y_pred,
                y_true=y_train,
                n_predictors=X_train.shape[1],
                name=str(self),
            )

        elif self._dataemitters is not None and self._dataemitter is not None:
            y_preds = []
            y_trues = []
            for emitter in self._dataemitters:
                (
                    X_train_df,
                    y_train_series,
                    X_test_df,
                    y_test_series,
                ) = emitter.emit_train_test_Xy()
                X_train = X_train_df.to_numpy()
                y_train = y_train_series.to_numpy()
                X_test = X_test_df.to_numpy()
                y_test = y_test_series.to_numpy()
                if verbose:
                    print_wrapped(f"Fitting {self._name}.", type="PROGRESS")
                self._estimator.fit(X_train, y_train)
                y_pred = self._estimator.predict(X_test)
                if y_scaler is not None:
                    y_pred = y_scaler.inverse_transform(y_pred)
                    y_test = y_scaler.inverse_transform(y_test)

                y_preds.append(y_pred)
                y_trues.append(y_test)

            self.cv_scorer = RegressionScorer(
                y_pred=y_preds,
                y_true=y_trues,
                n_predictors=X_train.shape[1],
                name=str(self),
            )

            # refit on all data
            X_train_df, y_train_series = self._dataemitter.emit_train_Xy()
            X_train = X_train_df.to_numpy()
            y_train = y_train_series.to_numpy()
            if verbose:
                print_wrapped(f"Fitting {self._name}.", type="PROGRESS")
            self._estimator.fit(X_train, y_train)
            y_pred = self._estimator.predict(X_train)
            if y_scaler is not None:
                y_pred = y_scaler.inverse_transform(y_pred)
                y_train = y_scaler.inverse_transform(y_train)

            self.train_scorer = RegressionScorer(
                y_pred=y_pred,
                y_true=y_train,
                n_predictors=X_train.shape[1],
                name=str(self),
            )

        else:
            raise ValueError("The datahandler must not be None")

        X_test_df, y_test_series = self._dataemitter.emit_test_Xy()
        X_test = X_test_df.to_numpy()
        y_test = y_test_series.to_numpy()

        y_pred = self._estimator.predict(X_test)
        if y_scaler is not None:
            y_pred = y_scaler.inverse_transform(y_pred)
            y_test = y_scaler.inverse_transform(y_test)

        self.test_scorer = RegressionScorer(
            y_pred=y_pred, y_true=y_test, n_predictors=X_test.shape[1], name=str(self)
        )

    def hyperparam_searcher(self):
        """Raises NotImplementedError. Not implemented for CustomR."""
        raise NotImplementedError("CustomR has no HyperparameterSearcher.")
