# TabularMagic
TabularMagic is a comprehensive wrapper of scikit-learn and statsmodels algorithms for rapid exploratory statistical and machine learning modeling of tabular data. TabularMagic reduces the headache of exploratory data analysis, regression analysis, and machine learning model benchmarking. Use TabularMagic to quickly explore your dataset, easily conduct regression analysis, and automatically compute baseline performance metrics across several popular machine learning models. 

Currently under development.

## Installation

TabularMagic can be installed via pip. The Python scripts below handle 
package setup and pip installation. 

To install TabularMagic: 
```
git clone https://github.com/ajy25/TabularMagic.git
python tm-install.py install
```

To uninstall TabularMagic:
```
python tm-install.py uninstall
```


## Getting started

We can build a TabularMagic object from a provided dataset. The 
TabularMagic object allows for rapid exploratory data analysis. 
```
import pandas as pd
import matplotlib.pyplot as plt
from tabularmagic import TabularMagic
from sklearn.datasets import load_diabetes

# load the dataset
diabetes_data = load_diabetes()
df = pd.DataFrame(data=diabetes_data.data, columns=diabetes_data.feature_names)
df['target'] = diabetes_data.target

# create a TabularMagic object
tm = TabularMagic(df, test_size=0.2, split_seed=42)
```

TabularMagic makes exploratory data analysis easy. Note that all TabularMagic 
plotting methods close figures before returning them for easier use with 
IPython notebooks. When working outside of IPython notebooks, returned figures 
must be reshown. 
```
def reshow(fig: plt.Figure):
    new_fig = plt.figure(figsize=fig.get_size_inches())
    new_manager = new_fig.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    plt.show()

train_eda = tm.eda()
print(train_eda.continuous_summary_statistics)
reshow(train_eda['target'].plot_distribution())
reshow(train_eda.plot_continuous_pairs(['target', 'age', 'bmi', 'bp']))
```

TabularMagic makes regression analysis easy via R-like formulas.
```
train_reg_report, test_reg_report = tm.lm_rlike('target ~ age + poly(bmi, 2) + exp(bp) + s1 * s2')
print(train_reg_report.statsmodels_summary())
reshow(train_reg_report.plot_diagnostics())
```

TabularMagic makes machine learning model benchmarking easy. Nested k-fold cross
validation handles hyperparameter selection and model evaluation on training 
data. The selected models are evaluated on the withheld testing data as well. 
Note that nested cross validation is computationally expensive and could take
some time to run. 
```
from tabularmagic.ml import Linear, TreeEnsemble, SVM
models =[
    Linear(),
    Linear(regularization_type='l1'),
    Linear(regularization_type='l2'),
    TreeEnsemble(ensemble_type='random_forest', n_jobs=-1),
    TreeEnsemble(ensemble_type='adaboost', n_jobs=-1),
    SVM(kernel='rbf', n_jobs=-1)
]
train_ml_report, test_ml_report = tm.ml_regression_benchmarking(
    X_vars=['age', 'bmi', 'bp', 's1', 's2'],
    y_var=['target'],
    models=models,  # 5-fold cross validation for hyperparameter search
    outer_cv=5      # 5-fold cross validation for model evaluation
)
print(train_ml_report.fit_statistics)
print(test_ml_report.fit_statistics)
reshow(test_ml_report['Linear(l2)'].plot_pred_vs_true())
```



## Demos

To learn more about TabularMagic functionality, check out the demos available in
the ./demo subdirectory. 








