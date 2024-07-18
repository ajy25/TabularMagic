from sklearn.base import BaseEstimator
from ....metrics.regression_scoring import RegressionScorer
from ..base_model import BaseDiscriminativeModel, HyperparameterSearcher
from ....data.datahandler import DataEmitter
from ....feature_selection import BaseFSR
from ....feature_selection.voteselect import VotingSelectionReport
from ....display.print_utils import print_wrapped


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
        self._feature_selectors = None
        self._max_n_features = 10
        self._name = "BaseR"
        self._train_scorer = None
        self.cv_scorer = None
        self._test_scorer = None
        self._voting_selection_report = None

        # By default, the first column is NOT dropped. For LinearR,
        # the first column is dropped to avoid multicollinearity.
        self._dropfirst = False

    def specify_data(
        self,
        dataemitter: DataEmitter | None = None,
        dataemitters: list[DataEmitter] | None = None,
        feature_selectors: list[BaseFSR] | None = None,
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
        feature_selectors : list[BaseFSR].
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
        if self._dataemitter is None:
            raise RuntimeError("DataEmitter not specified.")

        y_scaler = self._dataemitter.y_scaler()

        if self._dataemitters is None and self._dataemitter is not None:
            X_train_df, y_train_series = self._dataemitter.emit_train_Xy()

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

            y_train = y_train_series.to_numpy()

            self._hyperparam_searcher.fit(X_train, y_train, verbose=verbose)
            self._estimator = self._hyperparam_searcher._best_estimator
            y_pred = self._estimator.predict(X_train)
            if y_scaler is not None:
                y_pred = y_scaler.inverse_transform(y_pred)
                y_train = y_scaler.inverse_transform(y_train)
            self._train_scorer = RegressionScorer(
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
                y_train = y_train_series.to_numpy()
                y_test = y_test_series.to_numpy()

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

                self._hyperparam_searcher.fit(X_train, y_train, verbose=verbose)
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

            self._hyperparam_searcher.fit(X_train, y_train, verbose=verbose)
            self._estimator = self._hyperparam_searcher._best_estimator
            y_pred = self._estimator.predict(X_train)
            if y_scaler is not None:
                y_pred = y_scaler.inverse_transform(y_pred)
                y_train = y_scaler.inverse_transform(y_train)

            self._train_scorer = RegressionScorer(
                y_pred=y_pred,
                y_true=y_train,
                n_predictors=X_train.shape[1],
                name=str(self),
            )

        else:
            raise ValueError("DataEmitter or DataEmitters not specified.")

        X_test_df, y_test_series = self._dataemitter.emit_test_Xy()

        if self._feature_selectors is None:
            X_test = X_test_df.to_numpy()
        else:
            X_test = self._voting_selection_report._emit_test_X().to_numpy()

        y_test = y_test_series.to_numpy()

        y_pred = self._estimator.predict(X_test)
        if y_scaler is not None:
            y_pred = y_scaler.inverse_transform(y_pred)
            y_test = y_scaler.inverse_transform(y_test)

        self._test_scorer = RegressionScorer(
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

    def feature_selection_report(self) -> VotingSelectionReport:
        """Returns the VotingSelectionReport object.

        Returns
        -------
        VotingSelectionReport
        """
        if self._voting_selection_report is None:
            print_wrapped(
                f"No feature selection report available for {self._name}.",
                type="WARNING",
            )
        return self._voting_selection_report

    def _is_cross_validated(self) -> bool:
        """Returns True if the model is cross-validated.

        Returns
        -------
        bool
        """
        return self._dataemitters is not None

    def __str__(self):
        return self._name
