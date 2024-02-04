import pandas as pd
import matplotlib.pyplot as plt
plt.ioff()
from typing import Iterable
from ..metrics.regression_scoring import RegressionScorer
from ..models import *
from .viz import plot_predicted_vs_true_scatter


class RegressionReport():
    """An object that generates regression-relevant plots and tables for a 
    single model. 
    """

    def __init__(self, model: BaseRegression, X_test: pd.DataFrame, 
                 y_test: pd.DataFrame):
        """
        Initializes a RegressionReport object. 

        Parameters 
        ----------
        - model : BaseRegression
        - X_test : pd.DataFrame 
        - y_test : pd.DataFrame 

        Returns
        -------
        - None
        """
        self.model = model
        self._X_test_df = X_test
        self._y_test_df = y_test
        self._y_pred = self.model.predict(self._X_test_df.to_numpy())
        self._y_true = self._y_test_df.to_numpy().flatten()
        self.scorer = RegressionScorer(y_pred=self._y_pred, y_true=self._y_true, 
            n_regressors=self.model._n_regressors, model_id_str=str(self.model))
        
    def pred_vs_true_plot(self):
        """Returns a figure that is a scatter plot of the true and predicted y 
        values. 

        Parameters 
        ----------
        - None

        Returns
        -------
        - plt.Figure
        """
        fig = plot_predicted_vs_true_scatter(self._y_pred, self._y_true)
        fig.suptitle(f'{self._y_test_df.columns.to_list()[0]}: ' + \
                     f'Predicted vs True | ' + \
                     f'Pearson R = {round(self.scorer["pearsonr"], 3)}')
        fig.tight_layout()
        return fig

class ComprehensiveRegressionReport():
    """An object that generates regression-relevant plots and tables for a 
    set of models. Indexable. 
    """

    def __init__(self, models: Iterable[BaseRegression], X_test: pd.DataFrame, 
                 y_test: pd.DataFrame):
        """
        Initializes a ComprehensiveRegressionReport object. 

        Parameters 
        ----------
        - models : Iterable[BaseRegression]
        - X_test : pd.DataFrame 
        - y_test : pd.DataFrame 

        Returns
        -------
        - None
        """
        self.models = models
        self._X_test = X_test
        self._y_test = y_test
        self._report_dict_indexable_by_int = {
            i: RegressionReport(model, X_test, y_test) \
                for i, model in enumerate(self.models)}
        self._report_dict_indexable_by_str = {
            str(report.model): report for report in \
                self._report_dict_indexable_by_int.values()
        }
        self.fit_statistics = pd.concat([report.scorer.to_df() for report \
            in self._report_dict_indexable_by_int.values()], axis=1)

    def __getitem__(self, index: int | str) -> RegressionReport:
        """Indexes into ComprehensiveRegressionReport. 

        Parameters
        ----------
        - index : int | str. 

        Returns
        -------
        - RegressionReport. 
        """
        if isinstance(index, int):
            return self._report_dict_indexable_by_int[index]
        elif isinstance(index, str):
            return self._report_dict_indexable_by_str[index]
        else:
            raise ValueError(f'Invalid input: {index}.')


