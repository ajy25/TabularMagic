import numpy as np 
from sklearn.linear_model import LogisticRegression
from typing import Mapping, Literal, Iterable


from .base_classification import BaseClassification, HyperparameterSearcher



class LinearC(BaseClassification):
    """Logistic Regression classifier.

    Like all BaseClassification-derived classes, hyperparameter selection is
    performed automatically during training. The cross-validation and
    hyperparameter selection process can be modified by the user.
    """

    def __init__(self, type: Literal['no_penalty', 'l1', 'l2', 'elasticnet'] =\
                  'no_penalty',
                 hyperparam_search_method: Literal[None, 'grid', 'random'] = None,
                 hyperparam_grid_specification: Mapping[str, Iterable] = None,
                 model_random_state: int = 42,
                 name: str = None, **kwargs):
        """
        Initializes a LogisticRegressor object.

        Parameters
        ----------
        - type: Literal['l1', 'l2'].
            Default: 'l2'.
        - hyperparam_search_method: Literal[None, 'grid', 'random'].
            Default: None. If None, a classification-specific default hyperparameter
            search is conducted.
        - hyperparam_grid_specification: Mapping[str, Iterable].
            Default: None. If None, a classification-specific default hyperparameter
            search is conducted.
        - model_random_state: int.
            Default: 42. Random seed for the model.
        - name: str.
            Default: None. Determines how the model shows up in the reports.
            If None, the name is set to be the class name.
        - kwargs: Key word arguments are passed directly into the
            initialization of the HyperparameterSearcher class. In particular,
            inner_cv and inner_random_state can be set via kwargs.

        Notable kwargs
        --------------
        - inner_cv: int | BaseCrossValidator.
        - inner_cv_seed: int.
        - n_jobs: int. Number of parallel jobs to run.
        - verbose: int. sklearn verbosity level.
        """
        super().__init__()
        self._dropfirst = True

        if name is None:
            self._name = f'LogisticRegressor({type})'
        else:
            self._name = name


        if type == 'no_penalty':
            self._estimator = LogisticRegression(penalty=None,
                random_state=model_random_state)
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'fit_intercept': [True]
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )

        elif type == 'l1':
            self._estimator = LogisticRegression(penalty='l1',
                random_state=model_random_state,
                solver='liblinear')
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'C': np.logspace(-4, 4, 20)
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )

        elif type == 'l2':
            self._estimator = LogisticRegression(penalty='l2',
                random_state=model_random_state)
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'C': np.logspace(-4, 4, 20)
                }
            self._hyperparam_searcher = HyperparameterSearcher( 
                estimator=self._estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )

        elif type == 'elasticnet':
            self._estimator = LogisticRegression(penalty='elasticnet', 
                random_state=model_random_state,
                solver='saga')
            if (hyperparam_search_method is None) or \
                (hyperparam_grid_specification is None):
                hyperparam_search_method = 'grid'
                hyperparam_grid_specification = {
                    'C': np.logspace(-4, 4, 20),
                    'l1_ratio': np.linspace(0, 1, 20)
                }
            self._hyperparam_searcher = HyperparameterSearcher(
                estimator=self._estimator,
                method=hyperparam_search_method,
                grid=hyperparam_grid_specification,
                **kwargs
            )
        
        else:
            raise ValueError('Invalid value for type')




