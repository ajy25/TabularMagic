import statsmodels.api as sm
from ...metrics.regression_scoring import RegressionScorer
from ...data.datahandler import DataHandler


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
    

    def specify_data(self, datahandler: DataHandler, y_var: str, 
                     X_vars: list[str]):
        """Adds a DataHandler object to the model. 

        Parameters
        ----------
        - datahandler : DataHandler containing all data. Copy will be made
            for this specific model.
        - y_var : str. The name of the target variable.
        - X_vars : list[str]. The names of the predictor variables.
        """
        self._datahandler = datahandler.copy(y_var=y_var, X_vars=X_vars)


    def fit(self):
        """Fits the model based on the data specified.
        """
        y_scaler = self._datahandler.yscaler()

        X_train, y_train = self._datahandler.df_train_split(
            onehotted=True, dropfirst=True, dropna=True)
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

        X_test, y_test = self._datahandler.df_test_split(
            onehotted=True, dropfirst=True, dropna=True)
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



    def __str__(self):
        return self._name


