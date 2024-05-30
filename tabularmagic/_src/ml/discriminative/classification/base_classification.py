from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

from ..base_model import BaseDiscriminativeModel, HyperparameterSearcher
from ....data.datahandler import DataHandler
from ....metrics.classification_scoring import (
    ClassificationMulticlassScorer, ClassificationBinaryScorer)

import numpy as np



class BaseClassification(BaseDiscriminativeModel):
    """A class that provides the framework upon which all regression 
    objects are built. 

    BaseRegression wraps sklearn methods. 
    The primary purpose of BaseRegression is to automate the scoring and 
    model selection processes. 
    """

    def __init__(self):
        """Initializes a BaseRegression object. Creates copies of the inputs. 
        """
        self._label_encoder = LabelEncoder()
        self._hyperparam_searcher: HyperparameterSearcher = None
        self._estimator: BaseEstimator = None
        self._datahandler = None
        self._datahandlers = None
        self._name = 'BaseRegression'
        self.train_scorer = None
        self.train_overall_scorer = None
        self.test_scorer = None

        # By default, the first column is NOT dropped unless binary. For LinearR, 
        # the first column is dropped to avoid multicollinearity.
        self._dropfirst = False
        


    def specify_data(self, 
                     datahandler: DataHandler, 
                     datahandlers: list[DataHandler] = None):
        """Adds a DataHandler object to the model. 

        Parameters
        ----------
        - datahandler : DataHandler containing all data. Copy will be made
            for this specific model.
        - datahandlers : list[DataHandler]. 
            If not None, specifies the datahandlers for nested cross validation.
        """
        self._datahandler = datahandler
        self._datahandlers = datahandlers


    def fit(self):
        """Fits the model. Records training metrics, which can be done via 
        nested cross validation.
        """            
        is_binary = False

        if self._datahandlers is None and self._datahandler is not None:

            X_train_df, y_train_series = self._datahandler.df_train_split(
                dropfirst=self._dropfirst)
            X_train = X_train_df.to_numpy()
            y_train = y_train_series.to_numpy()


            y_train_encoded = self._label_encoder.fit_transform(y_train)


            if np.isin(np.unique(y_train_encoded), [0, 1]).all():
                is_binary = True

            self._hyperparam_searcher.fit(X_train, y_train_encoded)
            self._estimator = self._hyperparam_searcher._best_estimator

            y_pred = self._label_encoder.inverse_transform(
                self._estimator.predict(X_train)
            )


            if hasattr(self._estimator, 'predict_proba'):
                y_pred_score = self._estimator.predict_proba(X_train)
            elif hasattr(self._estimator, 'decision_function'):
                y_pred_score = self._estimator.decision_function(X_train)


            if not is_binary:
                self.train_scorer = ClassificationMulticlassScorer(
                    y_pred=y_pred,
                    y_true=y_train,
                    y_pred_score=y_pred_score,
                    y_pred_class_order=self._label_encoder.inverse_transform(
                        self._estimator.classes_),
                    name=str(self)
                )

            else:

                self.train_scorer = ClassificationBinaryScorer(
                    y_pred=y_pred,
                    y_true=y_train,
                    y_pred_score=y_pred_score,
                    name=str(self)
                )


        elif self._datahandlers is not None and self._datahandler is not None:
            y_preds = []
            y_trues = []
            y_pred_scores = []

            is_binary = True

            for datahandler in self._datahandlers:
                X_train_df, y_train_series = datahandler.df_train_split(
                    dropfirst=self._dropfirst)
                X_test_df, y_test_series = datahandler.df_test_split(
                    dropfirst=self._dropfirst)
                X_train = X_train_df.to_numpy()
                y_train = y_train_series.to_numpy()

                y_train_encoded: np.ndarray =\
                    self._label_encoder.fit_transform(y_train)

                if len(np.unique(y_train_encoded)) != 2:
                    is_binary = is_binary and False

                X_test = X_test_df.to_numpy()
                y_test = y_test_series.to_numpy()
                
                self._hyperparam_searcher.fit(X_train, y_train_encoded)
                fold_estimator = self._hyperparam_searcher._best_estimator

                y_pred = self._label_encoder.inverse_transform(
                    fold_estimator.predict(X_test))

                y_preds.append(y_pred)
                y_trues.append(y_test)

                if hasattr(fold_estimator, 'predict_proba'):
                    y_pred_scores.append(fold_estimator.predict_proba(X_test))
                elif hasattr(self._estimator, 'decision_function'):
                    y_pred_scores.append(
                        self._estimator.decision_function(X_test))


            if len(y_pred_scores) == 0:
                y_pred_scores = None

            if not is_binary:
                self.train_scorer = ClassificationMulticlassScorer(
                    y_pred=y_preds,
                    y_true=y_trues,
                    y_pred_score=y_pred_scores,
                    y_pred_class_order=self._label_encoder.\
                        inverse_transform(fold_estimator.classes_),
                    name=str(self)
                )
            else:
                self.train_scorer = ClassificationBinaryScorer(
                    y_pred=y_preds,
                    y_true=y_trues,
                    y_pred_score=y_pred_scores,
                    name=str(self)
                )
 
            # refit on all data
            X_train_df, y_train_series = self._datahandler.df_train_split(
                dropfirst=self._dropfirst)
            X_train = X_train_df.to_numpy()
            y_train = y_train_series.to_numpy()

            y_train_encoded = self._label_encoder.fit_transform(y_train)

            self._hyperparam_searcher.fit(X_train, y_train_encoded)
            self._estimator = self._hyperparam_searcher._best_estimator


            y_pred = self._label_encoder.inverse_transform(
                self._estimator.predict(X_train))
            if hasattr(fold_estimator, 'predict_proba'):
                y_pred_score = fold_estimator.predict_proba(X_train)
            elif hasattr(self._estimator, 'decision_function'):
                y_pred_score = self._estimator.decision_function(X_train)


            if not is_binary:
                self.train_overall_scorer = ClassificationMulticlassScorer(
                    y_pred=y_pred,
                    y_true=y_train,
                    y_pred_score=y_pred_score,
                    y_pred_class_order=self._label_encoder.\
                        inverse_transform(fold_estimator.classes_),
                    name=str(self)
                )
            else:
                self.train_overall_scorer = ClassificationBinaryScorer(
                    y_pred=y_pred,
                    y_true=y_train,
                    y_pred_score=y_pred_score,
                    name=str(self)
                )

        else:
            raise ValueError('The datahandler must not be None')

        X_test_df, y_test_series = self._datahandler.df_test_split(
            dropfirst=self._dropfirst)
        X_test = X_test_df.to_numpy()
        y_test = y_test_series.to_numpy()

        y_pred = self._label_encoder.inverse_transform(
            self._estimator.predict(X_test))


        y_pred_score = None
        if hasattr(self._estimator, 'predict_proba'):
            y_pred_score = self._estimator.predict_proba(X_test)
        elif hasattr(self._estimator, 'decision_function'):
            y_pred_score = self._estimator.decision_function(X_test)


        if not is_binary:
            self.test_scorer = ClassificationMulticlassScorer(
                y_pred=y_pred,
                y_true=y_test,
                y_pred_score=y_pred_score,
                y_pred_class_order=self._label_encoder.inverse_transform(
                    self._estimator.classes_),
                name=str(self)
            )

        else:
            self.test_scorer = ClassificationBinaryScorer(
                y_pred=y_pred,
                y_true=y_test,
                y_pred_score=y_pred_score,
                name=str(self)
            )


    def sklearn_estimator(self):
        """Returns the sklearn estimator object. 

        Returns
        -------
        - BaseEstimator
        """
        return self._estimator

    def hyperparam_searcher(self) -> HyperparameterSearcher:
        """Returns the HyperparameterSearcher object.
        
        Returns
        -------
        - HyperparameterSearcher
        """
        return self._hyperparam_searcher
    

    def _is_cross_validated(self) -> bool:
        """Returns True if the model is cross-validated. 

        Returns
        -------
        - bool
        """
        return self._datahandlers is not None


    def __str__(self):
        return self._name





