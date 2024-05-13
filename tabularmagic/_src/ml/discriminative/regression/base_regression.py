from sklearn.base import BaseEstimator
from ....metrics.regression_scoring import RegressionScorer
from ..base_model import BaseDiscriminativeModel, HyperparameterSearcher
from ....data.datahandler import DataHandler



class BaseRegression(BaseDiscriminativeModel):
    """A class that provides the framework upon which all regression 
    objects are built. 

    BaseRegression wraps sklearn methods. 
    The primary purpose of BaseRegression is to automate the scoring and 
    model selection processes. 
    """

    def __init__(self):
        """Initializes a BaseRegression object. Creates copies of the inputs. 
        """
        self._hyperparam_searcher: HyperparameterSearcher = None
        self._estimator: BaseEstimator = None
        self._datahandler = None
        self._datahandlers = None
        self._name = 'BaseRegression'
        self.train_scorer = None
        self.train_overall_scorer = None
        self.test_scorer = None

        # By default, the first column is NOT dropped. For LinearR, 
        # the first column is dropped to avoid multicollinearity.
        self._dropfirst = False


    def specify_data(self, datahandler: DataHandler, y_var: str, 
                     X_vars: list[str], 
                     outer_cv: int = None, 
                     outer_cv_seed: int = 42):
        """Adds a DataHandler object to the model. 

        Parameters
        ----------
        - datahandler : DataHandler containing all data. Copy will be made
            for this specific model.
        - y_var : str. The name of the target variable.
        - X_vars : list[str]. The names of the predictor variables.
        - outer_cv : int. Number of folds. 
            If None, does not conduct nested cross validation.
            Otherwise, conducts nested cross validation with outer_cv folds.
        - outer_cv_seed : int. Random seed for outer_cv.
        """
        self._datahandler = datahandler.copy(y_var=y_var, X_vars=X_vars)
        if outer_cv is not None:
            self._datahandlers = datahandler.kfold_copies(k=outer_cv, 
                y_var=y_var, X_vars=X_vars, seed=outer_cv_seed)


    def fit(self):
        """Fits the model. Records training metrics, which can be done via 
        nested cross validation.
        """
        y_scaler = self._datahandler.yscaler()

        if self._datahandlers is None and self._datahandler is not None:
            X_train_df, y_train_series = self._datahandler.df_train_split(
                dropfirst=self._dropfirst)
            X_train = X_train_df.to_numpy()
            y_train = y_train_series.to_numpy()
            self._hyperparam_searcher.fit(X_train, y_train)
            self._estimator = self._hyperparam_searcher.best_estimator

            y_pred = self._estimator.predict(X_train)
            if y_scaler is not None:
                y_pred = y_scaler.inverse_transform(y_pred)
                y_train = y_scaler.inverse_transform(y_train)

            self.train_scorer = RegressionScorer(
                y_pred=y_pred,
                y_true=y_train,
                n_predictors=X_train.shape[1],
                name=str(self) + '_train'
            )

        elif self._datahandlers is not None and self._datahandler is not None:
            y_preds = []
            y_trues = []
            for datahandler in self._datahandlers:
                X_train_df, y_train_series = datahandler.df_train_split(
                    dropfirst=self._dropfirst)
                X_test_df, y_test_series = datahandler.df_test_split(
                    dropfirst=self._dropfirst)
                X_train = X_train_df.to_numpy()
                y_train = y_train_series.to_numpy()
                X_test = X_test_df.to_numpy()
                y_test = y_test_series.to_numpy()
                self._hyperparam_searcher.fit(X_train, y_train)
                fold_estimator = self._hyperparam_searcher.best_estimator

                y_pred = fold_estimator.predict(X_test)
                if y_scaler is not None:
                    y_pred = y_scaler.inverse_transform(y_pred)
                    y_test = y_scaler.inverse_transform(y_test)

                y_preds.append(y_pred)
                y_trues.append(y_test)

            self.train_scorer = RegressionScorer(
                y_pred=y_preds,
                y_true=y_trues,
                n_predictors=X_train.shape[1],
                name=str(self) + '_train_cv'
            )

            # refit on all data
            X_train_df, y_train_series = self._datahandler.df_train_split(
                dropfirst=self._dropfirst)
            X_train = X_train_df.to_numpy()
            y_train = y_train_series.to_numpy()
            self._hyperparam_searcher.fit(X_train, y_train)
            self._estimator = self._hyperparam_searcher.best_estimator
            y_pred = self._estimator.predict(X_train)
            if y_scaler is not None:
                y_pred = y_scaler.inverse_transform(y_pred)
                y_train = y_scaler.inverse_transform(y_train)

            self.train_overall_scorer = RegressionScorer(
                y_pred=y_pred,
                y_true=y_train,
                n_predictors=X_train.shape[1],
                name=str(self) + '_train_no_cv'
            )


        else:
            raise ValueError('Error occured in fit method.')

        X_test_df, y_test_series = self._datahandler.df_test_split(
            dropfirst=self._dropfirst)
        X_test = X_test_df.to_numpy()
        y_test = y_test_series.to_numpy()

        y_pred = self._estimator.predict(X_test)
        if y_scaler is not None:
            y_pred = y_scaler.inverse_transform(y_pred)
            y_test = y_scaler.inverse_transform(y_test)
        
        self.test_scorer = RegressionScorer(
            y_pred=y_pred,
            y_true=y_test,
            n_predictors=X_train.shape[1],
            name=str(self) + '_test'
        )

    def sklearn_estimator(self):
        """Returns the sklearn estimator object. 

        Returns
        -------
        - BaseEstimator
        """
        return self._estimator


    def __str__(self):
        return self._name





