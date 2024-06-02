import statsmodels.api as sm
from typing import Literal
from ...metrics.regression_scoring import RegressionScorer
from ...data.datahandler import DataEmitter


class OrdinaryLeastSquares:
    """Statsmodels OLS wrapper.
    """

    def __init__(self, name: str = None):
        """
        Initializes a OrdinaryLeastSquares object. Regresses y on X.

        Parameters
        ----------
        - nickname : str. 
            Default: None. Determines how the model shows up in the reports. 
            If None, the nickname is set to be the class name.
        """
        self.estimator = None
        self._name = name
        if self._name is None:
            self._name = f'OrdinaryLeastSquares'
    

    def specify_data(self, dataemitter: DataEmitter):
        """Adds a DataHandler object to the model. 

        Parameters
        ----------
        - datahandler : DataHandler containing all data. X and y variables 
            must be specified.
        """
        self._dataemitter = dataemitter


    def fit(self):
        """Fits the model based on the data specified.
        """
        y_scaler = self._dataemitter.y_scaler()

        X_train, y_train = self._dataemitter.emit_train_Xy()
        n_predictors = X_train.shape[1]
        X_train = sm.add_constant(X_train)
        self.estimator = sm.OLS(y_train, X_train).fit(cov_type='HC3')
        

        y_pred_train = self.estimator.predict(X_train).to_numpy()
        if y_scaler is not None:
            y_pred_train = y_scaler.inverse_transform(y_pred_train)
            y_train = y_scaler.inverse_transform(y_train)


        self.train_scorer = RegressionScorer(
            y_pred=y_pred_train,
            y_true=y_train.to_numpy(),
            n_predictors=n_predictors,
            name=self._name + '_train'
        )

        X_test, y_test = self._dataemitter.emit_test_Xy()
        X_test = sm.add_constant(X_test)


        y_pred_test = self.estimator.predict(X_test).to_numpy()
        if y_scaler is not None:
            y_pred_test = y_scaler.inverse_transform(y_pred_test)
            y_test = y_scaler.inverse_transform(y_test)


        self.test_scorer = RegressionScorer(
            y_pred=y_pred_test,
            y_true=y_test.to_numpy(),
            n_predictors=n_predictors,
            name=self._name + '_test'
        )


    
    def _backwards_elimination(self, 
            criteria: Literal['aic'] = 'aic') -> list[str]:
        """Performs backwards elimination on the train dataset to identify a
        subset of predictors that are most likely to be significant. 
        Returns only the subset of predictors identified.

        Categorical variables will either be included or excluded as a whole.

        Parameters
        ----------
        - criteria : str. Default: 'aic'.

        Returns
        -------
        - list of str. 
            The subset of predictors that are most likely to be significant.
        """
        raise NotImplementedError(
            'Backwards elimination is not yet implemented.')



    def __str__(self):
        return self._name


