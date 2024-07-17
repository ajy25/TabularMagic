import statsmodels.api as sm
import numpy as np
from ..metrics.regression_scoring import RegressionScorer
from ..data.datahandler import DataEmitter


class NegativeBinomialLinearModel:
    """Statsmodels GLM wrapper for the negative binomial family"""

    def __init__(self, name: str | None = None):
        """
        Initializes a NegativeBinomialLinearModel object. Regresses y on X.

        Parameters
        ----------
        name : str.
            Default: None. Determines how the model shows up in the reports.
            If None, the name is set to be the class name.
        """
        self.estimator = None
        self._name = name
        if self._name is None:
            self._name = "Negative Binomial GLM"

    def specify_data(self, dataemitter: DataEmitter):
        """Adds a DataEmitter object to the model.

        Parameters
        ----------
        dataemitter : DataEmitter containing all data. X and y variables
            must be specified.
        """
        self._dataemitter = dataemitter

    def fit(self):
        """Fits the model based on the data specified."""

        # Emit the training data
        X_train, y_train = self._dataemitter.emit_train_Xy()
        # Add a constant to the Design Matrix
        X_train = sm.add_constant(X_train)

        # Set the estimator to be a generalized linear model with a log link
        self.estimator = sm.GLM(
            y_train, X_train, family=sm.families.NegativeBinomial()
        ).fit(cov_type="HC3")

        # Get the predictions from the training dataset
        y_pred_train: np.ndarray = self.estimator.predict(exog=X_train).to_numpy()

        # Emit the test data
        X_test, y_test = self._dataemitter.emit_test_Xy()
        X_test = sm.add_constant(X_test)

        n_predictors = X_train.shape[1]

        self.train_scorer = RegressionScorer(
            y_pred=y_pred_train,
            y_true=y_train.to_numpy(),
            n_predictors=n_predictors,
            name=self._name,
        )

        y_pred_test = self.estimator.predict(X_test).to_numpy()
        self.test_scorer = RegressionScorer(
            y_pred=y_pred_test,
            y_true=y_test.to_numpy(),
            n_predictors=n_predictors,
            name=self._name,
        )

        def __str__(self):
            return self._name
