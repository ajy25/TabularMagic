import statsmodels.api as sm
from typing import Literal
from ...metrics.regression_scoring import RegressionScorer
from ...data.datahandler import DataEmitter

class GLM:

    """Statsmodels GLM wrapper
    """

    def __init__(self, name: str = None):
        """
        Initializes a GLM object. Regresses y on X.

        Parameters
        ----------
        name : str. 
            Default: None. Determines how the model shows up in the reports. 
            If None, the name is set to be the class name.
        """
        self.estimator = None
        self._name = name
        if self._name is None:
            self._name = 'GLM'
    
    def specify_data(self, dataemitter: DataEmitter):
        """Adds a DataEmitter object to the model. 

        Parameters
        ----------
        dataemitter : DataEmitter containing all data. X and y variables 
            must be specified.
        """
        self._dataemitter = dataemitter

    def fit(self, family: Literal['binomial', 'gamma', 'gaussian', 'poisson']):
        """Fits the model based on the data specified.

        Parameters
        ----------
        family : Literal['binomial', 'gamma','gaussian','poisson']
            Specifies the family of Distributions
        """

        X_train, y_train = self._dataemitter.emit_train_Xy()
        n_predictors = X_train.shape[1]
        X_train = sm.add_constant(X_train)

        # Fit the model depending on the family and link function chosen 
        if(family == 'binomial'):
            self.estimator = sm.families.Binomial(link=sm.families.links.logit())
        elif(True):
            raise NotImplementedError(
            'Family not yet implemented / does not exist')

        y_pred_train = self.estimator.predict(X_train).to_numpy()

        self.train_scorer = RegressionScorer(
            y_pred=y_pred_train,
            y_true=y_train.to_numpy(),
            n_predictors=n_predictors,
            name=self._name
        )

        X_test, y_test = self._dataemitter.emit_test_Xy()
        X_test = sm.add_constant(X_test)


        y_pred_test = self.estimator.predict(X_test).to_numpy()


        self.test_scorer = RegressionScorer(
            y_pred=y_pred_test,
            y_true=y_test.to_numpy(),
            n_predictors=n_predictors,
            name=self._name
        )

