import statsmodels.api as sm
from sklearn.metrics import f1_score
import numpy as np
from ..metrics.classification_scoring import ClassificationBinaryScorer
from ..data.datahandler import DataEmitter


class BinomialLinearModel:
    """Statsmodels GLM wrapper for the Binomial family"""

    def __init__(self, name: str | None = None):
        """
        Initializes a BinomialLinearModel object. Regresses y on X.

        Parameters
        ----------
        name : str.
            Default: None. Determines how the model shows up in the reports.
            If None, the name is set to be the class name.
        """
        self.estimator = None
        self._name = name
        if self._name is None:
            self._name = "Binomial GLM"

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

        X_train, y_train = self._dataemitter.emit_train_Xy()
        X_train = sm.add_constant(X_train)

        self.estimator = sm.GLM(
            y_train,
            X_train,
            family=sm.families.Binomial(link=sm.families.links.Logit()),
        ).fit(cov_type="HC3")

        y_pred_train: np.ndarray = self.estimator.predict(exog=X_train).to_numpy()

        X_test, y_test = self._dataemitter.emit_test_Xy()
        X_test = sm.add_constant(X_test)

        # Find the best threshold based on f1 score
        best_score = None
        best_threshold = None
        for temp_threshold in np.linspace(0.0, 1.0, num=21):
            y_pred_train_binary = (y_pred_train > temp_threshold).astype(int)
            curr_score = f1_score(y_train, y_pred_train_binary)
            if best_score is None or curr_score > best_score:
                best_score = curr_score
                best_threshold = temp_threshold

        # print(f"Threshold found: {best_threshold}. F1 Score: {best_score}")

        y_pred_train_binary = (y_pred_train >= best_threshold).astype(int)

        self.train_scorer = ClassificationBinaryScorer(
            y_pred=y_pred_train_binary,
            y_true=y_train.to_numpy(),
            y_pred_score=np.hstack(
                [
                    np.zeros(shape=(len(y_pred_train), 1)),
                    y_pred_train.reshape(-1, 1),
                ]
            ),
            name=self._name,
        )

        y_pred_test = self.estimator.predict(X_test).to_numpy()
        y_pred_test_binary = (y_pred_test >= best_threshold).astype(int)

        self.test_scorer = ClassificationBinaryScorer(
            y_pred=y_pred_test_binary,
            y_true=y_test.to_numpy(),
            y_pred_score=np.hstack(
                [np.zeros(shape=(len(y_pred_test), 1)), y_pred_test.reshape(-1, 1)]
            ),
            name=self._name,
        )

        def __str__(self):
            return self._name
