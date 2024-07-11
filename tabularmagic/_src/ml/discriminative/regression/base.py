from sklearn.base import BaseEstimator
from ....metrics.regression_scoring import RegressionScorer
from ..base_model import BaseDiscriminativeModel, HyperparameterSearcher
from ....data.datahandler import DataEmitter


class BaseR(BaseDiscriminativeModel):
    """Class that provides the framework that all TabularMagic regression
    classes inherit.

    The primary purpose of BaseR is to automate the scoring and
    model selection processes.
    """

    def __init__(self):
        """Initializes a BaseR object."""
        self._hyperparam_searcher: HyperparameterSearcher = None
        self._estimator: BaseEstimator = None
        self._dataemitter = None
        self._dataemitters = None
        self._name = "BaseR"
        self.train_scorer = None
        self.cv_scorer = None
        self.test_scorer = None

        # By default, the first column is NOT dropped. For LinearR,
        # the first column is dropped to avoid multicollinearity.
        self._dropfirst = False

    def specify_data(
        self, dataemitter: DataEmitter, dataemitters: list[DataEmitter] | None = None
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
        y_scaler = self._dataemitter.y_scaler()

        if self._dataemitters is None and self._dataemitter is not None:
            X_train_df, y_train_series = self._dataemitter.emit_train_Xy()
            X_train = X_train_df.to_numpy()
            y_train = y_train_series.to_numpy()
            self._hyperparam_searcher.fit(X_train, y_train)
            self._estimator = self._hyperparam_searcher._best_estimator
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
                self._hyperparam_searcher.fit(X_train, y_train)
                fold_estimator = self._hyperparam_searcher._best_estimator

                y_pred = fold_estimator.predict(X_test)
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
            self._hyperparam_searcher.fit(X_train, y_train)
            self._estimator = self._hyperparam_searcher._best_estimator
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
            raise ValueError("DataEmitter or DataEmitters not specified.")

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

    def sklearn_estimator(self) -> BaseEstimator:
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

    def _is_cross_validated(self) -> bool:
        """Returns True if the model is cross-validated.

        Returns
        -------
        bool
        """
        return self._dataemitters is not None

    def __str__(self):
        return self._name
