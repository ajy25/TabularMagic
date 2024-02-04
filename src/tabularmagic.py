import pandas as pd


class TabularMagic():
    """TabularMagic: Automatic statistical and machine learning analysis of 
    datasets in tabular form.
    """

    def __init__(self, df: pd.DataFrame, models: list[]):
        """Initializes a TabularMagic object.

        Parameters
        ----------
        - df : pd.DataFrame ~ (n_samples, n_variables)

        Returns
        -------
        - None
        """
        self.original_df = df.copy()
        self.shape = self.original_df.shape
        self._X_vars = []
        self._y_var = []

    def set_models():
        """
        
        """
        


    



