import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, BaseCrossValidator
import os
from joblib import dump
from ....metrics.classification_scoring import ClassificationScorer
from ..base_model import BaseDiscriminativeModel, HyperparameterSearcher


class BaseClassification(BaseDiscriminativeModel):
    """A class that provides the framework upon which all classification 
    classes are built. 

    BaseClassification wraps sklearn methods. 
    The primary purpose of BaseClassification is to automate the scoring and 
    model selection processes. 
    """

    def __init__(self, X: np.ndarray = None, y: np.ndarray = None):
        """Initializes a BaseClassification object. 
        Creates copies of the inputs. 

        Parameters
        ----------
        - X : np.ndarray ~ (n_examples, n_predictors).
            Default: None. Matrix of predictor variables. 
        - y : np.ndarray ~ (n_examples).
            Default: None. Dependent variable vector. 

        Returns
        -------
        - None
        """
        self._hyperparam_searcher: HyperparameterSearcher = None
        self.estimator: BaseEstimator = None
        if (X is not None) and (y is not None):
            self._X = X.copy()
            self._y = y.copy()
            self._n_samples = X.shape[0]
            self._n_predictors = X.shape[1]
        self.nickname = 'BaseClassification'

    def fit(self, X: np.ndarray = None, y: np.ndarray = None, 
            outer_cv: int | BaseCrossValidator | None = None,
            outer_cv_seed: int = 42):
        """Fits the model. Records training metrics, which can be done via 
        nested cross validation.

        Parameters
        ----------
        - X : np.ndarray ~ (n_samples, n_predictors).
            Default: None.
        - y : np.ndarray ~ (n_samples).
            Default: None.
        - outer_cv : int | BaseCrossValidator | None. 
            Default: None. If None, does not conduct nested cross validaiton. 
            In this case, the train scores are computed over the entire training
            dataset, over which the model has been fitted. 
        - outer_cv_seed : int.
            Random state of the outer cross validator.

        Returns
        -------
        - None
        """
        if (X is not None) and (y is not None):
            self._X = X.copy()
            self._y = y.copy()
            self._n_samples = X.shape[0]
            self._n_regressors = X.shape[1]
        if ((self._X is None) or (self._y is None)) and \
            (not ((self._X is None) and (self._y is None))):
            raise ValueError(f'Invalid input: X, y.',
                             'X and y both must not be None.')
        self._verify_Xy_input_validity(self._X, self._y)

        if outer_cv is None:
            self._hyperparam_searcher.fit(self._X, self._y)
            self.estimator = self._hyperparam_searcher.best_estimator
            self.train_scorer = ClassificationScorer(
                y_pred=self.estimator.predict(self._X),
                y_true=self._y,
                n_regressors=self._n_regressors,
                model_id_str=str(self)
            )
        else:
            if isinstance(outer_cv, int):
                cv = KFold(n_splits=outer_cv,
                           shuffle=True, random_state=outer_cv_seed)
            elif isinstance(outer_cv, BaseCrossValidator):
                cv = outer_cv
            y_preds = []
            y_trues = []
            for train_index, test_index in cv.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                self._hyperparam_searcher.fit(X_train, y_train)
                fold_estimator = self._hyperparam_searcher.best_estimator
                y_preds.append(fold_estimator.predict(X_test))
                y_trues.append(y_test)
            self.train_scorer = ClassificationScorer(
                y_pred=y_preds,
                y_true=y_trues,
                n_regressors=self._n_regressors,
                model_id_str=str(self)
            )
            self._hyperparam_searcher.fit(self._X, self._y)
            self.estimator = self._hyperparam_searcher.best_estimator

    def _verify_Xy_input_validity(self, X: np.ndarray, y: np.ndarray):
        """Verifies that the inputs X and y are valid. If invalid, raises 
        the appropriate error with the appropriate error message. 

        Parameters
        ----------
        - X : np.ndarray ~ (n_samples, n_regressors).
        - y : np.ndarray ~ (n_samples).

        Returns
        -------
        - None
        """
        if not isinstance(X, np.ndarray):
            raise ValueError(f'Invalid input: X. Must be 2d np array.')
        if not isinstance(y, np.ndarray):
            raise ValueError(f'Invalid input: y. Must be 1d np array.')
        if len(X.shape) != 2:
            raise ValueError(f'Invalid input: X. Must be 2d np array.')
        if len(y.shape) != 1:
            raise ValueError(f'Invalid input: y. Must be 1d np array.')
        if X.shape[0] != y.shape[0]:
            raise ValueError(f'Invalid input: X, y. Must have the same',
                             'length in the first dimension.')

    def _verify_X_input_validity(self, X: np.ndarray):
        """Verifies that the input X is valid. If invalid, raises 
        the appropriate error with the appropriate error message. 

        Parameters
        ----------
        - X : np.ndarray ~ (n_samples, n_regressors).

        Returns
        -------
        - None
        """
        if not isinstance(X, np.ndarray):
            raise ValueError(f'Invalid input: X. Must be 2d np array.')
        if X.shape[1] != self._n_regressors:
            raise ValueError(f'Invalid input: X. Must have the same',
                             'length in the second dimension as the dataset',
                             'upon which the estimator has been trained.')
        
    def save_checkpoint(self, dirpath: str = None, filepath: str = None):
        """Saves the best estimator to a joblib file.

        To load model, run the following code:
        from joblib import load
        load(filepath)
        
        Parameters
        ----------
        - dirpath : str. If None, defers to filepath. Otherwise, saves default
            name file into directory specified by dirpath.
        - filepath : str. If both filepath and dirpath are None, saves to 
            current working directory with default name. 
        """
        default_name = self.nickname + '.joblib'
        if dirpath is not None:
            dump(self.estimator, os.path.join(dirpath, default_name))
        elif filepath is not None:
            dump(self.estimator, filepath)
        else:
            dump(self.estimator, os.path.join(os.getcwd(), default_name))

    def __str__(self):
        return self.nickname
