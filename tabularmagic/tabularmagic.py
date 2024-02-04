import pandas as pd
from typing import Iterable
from sklearn.model_selection import train_test_split
from .models import *
from .visualization import ComprehensiveRegressionReport

class TabularMagic():
    """TabularMagic: Automatic statistical and machine learning analysis of 
    datasets in tabular form.
    """

    def __init__(self, df: pd.DataFrame):
        """Initializes a TabularMagic object. 
        
        Note: DataFrame indices are not guaranteed to be correctly preserved. 

        Parameters
        ----------
        - df : pd.DataFrame ~ (n_samples, n_variables)

        Returns
        -------
        - None
        """
        self.original_df = df.copy()
        self.shape = self.original_df.shape

    def comprehensive_model_benchmarking(self, X_vars: list[str], y_var: str, 
                                   models: Iterable[BaseModel], 
                                   test_size: float = 0.2, 
                                   random_state: int = 42):
        """Conducts a comprehensive benchmarking exercise. 

        Parameters
        ----------
        - X_vars : list[str]. 
        - y_var : str. 
        - models : Iterable[BaseModel]. 
            Testing performance of all models will be evaluated. 
        - preprocessing_steps: Iterable[str]. 
        - test_size : float. 
            Default: 0.2. Proportion of the dataset to withhold. 
        - random_state : int.
            Default: 42. 

        Returns
        -------
        - report : ComprehensiveRegressionReport.
        - train_metrics : pd.DataFrame. 
            Output provides statistics for each model on the train dataset. 
        - test_metrics : pd.DataFrame. 
            Output provides statistics for each model on the test dataset. 
        """
        self._X_vars = X_vars
        self._y_var = y_var
        self._models = models
        self._test_size = test_size

        X_df_subset = self.original_df[X_vars]
        y_df_subset = self.original_df[y_var]
        X_train, X_test, y_train, y_test = train_test_split(X_df_subset, 
            y_df_subset, test_size=test_size, random_state=random_state)
        self._X_train_df = pd.DataFrame(X_train, columns=X_vars)
        self._X_test_df = pd.DataFrame(X_test, columns=X_vars)
        self._y_train_df = pd.DataFrame(y_train, columns=y_var)
        self._y_test_df = pd.DataFrame(y_test, columns=y_var)

        for i, model in enumerate(models):
            print(f'Task {i+1} of {len(models)}. \t Training {model}.')
            model.fit(self._X_train_df.to_numpy(), 
                      self._y_train_df.to_numpy().flatten())

        train_report = ComprehensiveRegressionReport(models, X_train, y_train)
        test_report = ComprehensiveRegressionReport(models, X_test, y_test)

        return train_report, test_report


        


    



