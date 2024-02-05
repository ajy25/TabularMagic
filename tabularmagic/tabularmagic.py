import pandas as pd
from typing import Iterable
from sklearn.model_selection import train_test_split
from .models import *
from .visualization import ComprehensiveRegressionReport
from .preprocessing import DataPreprocessor


class TabularMagic():
    """TabularMagic: Automatic statistical and machine learning analysis of 
    datasets in tabular form.
    """

    def __init__(self, df: pd.DataFrame, df_test: pd.DataFrame = None, 
                test_size: float = 0.2, random_state: int = 42):
        """Initializes a TabularMagic object. 
        
        Note: DataFrame indices are not guaranteed to be correctly preserved. 

        Parameters
        ----------
        - df : pd.DataFrame ~ (n_samples, n_variables).
        - df_test : pd.DataFrame ~ (n_test_samples, n_variables).
            Default: None. If not None, then treats df as the training 
            DataFrame. 
        - test_size : float. 
            Default: 0.2. Proportion of the dataset to withhold for 
            testing. If df_test is provided, then test_size is
            ignored. 
        - random_state : int.
            Default: 42. Used for train test split. 
            If df_test is provided, then random_state is ignored. 

        Returns
        -------
        - None
        """
        self.original_df = df.copy()
        if df_test is not None:
            self.original_df_train = self.original_df
            self.original_df_test = df_test.copy()
        else:
            temp_train, temp_test = train_test_split(self.original_df, 
                test_size=test_size, shuffle=True, random_state=random_state)
            self.original_df_train = pd.DataFrame(temp_train, 
                columns=df.columns)
            self.original_df_test = pd.DataFrame(temp_test, columns=df.columns)
        self.working_df_train = self.original_df_train.copy()
        self.working_df_test = self.original_df_test.copy()


    def comprehensive_eda(self): 
        pass


    def preprocess_data(self, ):
        pass


    def comprehensive_model_benchmarking(self, X_vars: list[str], y_var: str, 
                                   models: Iterable[BaseModel]):
        """Conducts a comprehensive benchmarking exercise. 

        Parameters
        ----------
        - X_vars : list[str]. 
        - y_var : str. 
        - models : Iterable[BaseModel]. 
            Testing performance of all models will be evaluated. 

        Returns
        -------
        - train_report : ComprehensiveRegressionReport.
        - test_report : ComprehensiveRegressionReport.
        """
        self._X_vars = X_vars
        self._y_var = y_var
        self._models = models

        local_X_train_df = self.working_df_train[X_vars]
        local_X_test_df = self.working_df_test[X_vars]
        local_y_train_df = self.working_df_train[y_var]
        local_y_test_df = self.working_df_test[y_var]

        X_train_np = local_X_train_df.to_numpy()
        y_train_np = local_y_train_df.to_numpy().flatten()

        for i, model in enumerate(models):
            print(f'Task {i+1} of {len(models)}. \t Training {model}.')
            model.fit(X_train_np, y_train_np)

        train_report = ComprehensiveRegressionReport(
            models, local_X_train_df, local_y_train_df)
        test_report = ComprehensiveRegressionReport(
            models, local_X_test_df, local_y_test_df)

        return train_report, test_report


        


    



