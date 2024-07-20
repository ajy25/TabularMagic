import statsmodels.api as sm
import numpy as np
from ..metrics.regression_scoring import RegressionScorer
from ..data.datahandler import DataEmitter
from scipy.stats import chi2


class CountLinearModel:
    """Statsmodels GLM wrapper that automatically chooses a Poisson or
    Negative Binomial GLM based on a likelihood ratio test for overdispersion
    """

    def __init__(self, name: str | None = None):
        """
        Initializes a CountLinearModel object. Regresses y on X.

        Parameters
        ----------
        name : str.
            Default: None. Determines how the model shows up in the reports.
            If None, the name is set to be the class name.
        """
        self.estimator = None
        self._name = name
        self._type = None
        if self._name is None:
            self._name = "Count GLM"

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
        y_scaler = self._dataemitter.y_scaler()

        # Emit the training data
        X_train, y_train = self._dataemitter.emit_train_Xy()
        # Add a constant to the Design Matrix
        X_train = sm.add_constant(X_train)

        # Fit Poisson and ngative binomial glm and perform likelihood ratio test
        poisson_estimator = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit(
            cov_type="HC3"
        )

        negbin_estimator = sm.NegativeBinomial(y_train, X_train).fit(cov_type="HC3")

        # Extract log-likelihoods
        poisson_llf = poisson_estimator.llf
        negbin_llf = negbin_estimator.llf

        # Based off of odTest from pscl package in r
        d = 2 * (negbin_llf - poisson_llf)
        pval = (1 - chi2.cdf(d, df=1)) / 2
        alpha = 0.05

        # test
        print(f"pval: {pval}")

        # H0: Poisson; Ha: Negative Binomial
        if pval <= alpha:
            self.estimator = negbin_estimator
            self._type = "negativebinomial"
        else:
            self.estimator = poisson_estimator
            self._type = "poisson"

        # Get the predictions from the training dataset
        y_pred_train: np.ndarray = self.estimator.predict(exog=X_train).to_numpy()
        if y_scaler is not None:
            y_pred_train = y_scaler.inverse_transform(y_pred_train)
            y_train = y_scaler.inverse_transform(y_train)


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
        if y_scaler is not None:
            y_pred_test = y_scaler.inverse_transform(y_pred_test)
            y_test = y_scaler.inverse_transform(y_test)


        self.test_scorer = RegressionScorer(
            y_pred=y_pred_test,
            y_true=y_test.to_numpy(),
            n_predictors=n_predictors,
            name=self._name,
        )

        def __str__(self):
            return self._name
