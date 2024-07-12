# TabularMagic

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

TabularMagic is a Python package for rapid exploratory statistical and machine learning modeling of wide format tabular data. TabularMagic empowers users to quickly explore new datasets, conduct regression analyses with ease, and effortlessly compute baseline performance metrics across a wide range of popular machine learning models. TabularMagic excels in handling datasets with fewer than 10,000 examples. 


### Why does TabularMagic exist?

Though numerous auto-ML solutions have emerged to streamline data science workflows at an enterprise scale, low-code data science packages tailored for small tabular datasets remain scarce. TabularMagic strives to fill this void, offering a straightforward Python interface for common data science routines. This package relieves users from the tedious tasks often associated with such projects – maintaining separate train and test data, one-hot encoding and scaling features, and proper cross-validation benchmarking of various machine learning models, many of which require hyperparameter tuning.

## Installation and dependencies

TabularMagic can be installed via pip. The Python scripts below handle 
package setup and pip installation. 

To install TabularMagic: 
```bash
git clone https://github.com/ajy25/TabularMagic.git
cd tabularmagic
python tmbuild.py install
```

To uninstall TabularMagic:
```bash
python tmbuild.py uninstall
```

TabularMagic is built on top of the Python data science stack. For additional notes regarding dependencies, check out `./dev_notes/dependencies.md`. TabularMagic requires Python version 3.10 or later.

## Getting started

### Example Python API usage

We can build an Analyzer object on top of a given dataset.
```python
import pandas as pd
import matplotlib.pyplot as plt
import tabularmagic as tm
from sklearn.datasets import load_diabetes

# load the dataset
diabetes_data = load_diabetes()
df = pd.DataFrame(data=diabetes_data.data, columns=diabetes_data.feature_names)
df["target"] = diabetes_data.target

# create a Analyzer object
analyzer = tm.Analyzer(df, test_size=0.2, split_seed=42)
print(analyzer)
```

TabularMagic makes exploratory data analysis easy. Note that all TabularMagic plotting methods close figures before returning them for easier use with IPython notebooks. When working outside of IPython notebooks, returned figures must be reshown. 
```python
def reshow(fig: plt.Figure):
    new_fig = plt.figure(figsize=fig.get_size_inches())
    new_manager = new_fig.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    plt.show()

train_eda = analyzer.eda()
print(train_eda.numerical_summary_statistics())
reshow(train_eda.plot_distribution("target"))
reshow(train_eda.plot_numerical_pairs(["target", "age", "bmi", "bp"]))
```

TabularMagic makes regression analysis easy.
```python
lm_report = analyzer.lm(
    formula="target ~ age + bmi"
)
lm_report.statsmodels_summary()
lm_report.train_report().set_outlier_threshold(2).plot_diagnostics(
    show_outliers=True)
```

TabularMagic makes machine learning model benchmarking easy. Nested k-fold cross validation handles hyperparameter selection and model evaluation on training data. The selected models are then further evaluated on the withheld testing data. Note that nested cross validation is computationally expensive and could take some time to run; to disable nested cross validation (i.e., only compute train and test fit statistics), simply set `outer_cv = None`.
```python
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV

models = [
    tm.ml.LinearR(),
    tm.ml.LinearR("l1"),
    tm.ml.LinearR("l2"),
    tm.ml.TreeEnsembleR("random_forest", n_jobs=-1),
    tm.ml.TreeEnsembleR("adaboost", n_jobs=-1),
    tm.ml.SVMR("rbf", n_jobs=-1),
    tm.ml.CustomR(
        estimator=Pipeline(
            steps=[
                ('feature_selection', SelectKBest(k=2)),
                ('regression', GridSearchCV(
                    estimator=Lasso(alpha=0.5),
                    param_grid={'alpha': np.logspace(-4, 4, 10)}
                ))
            ]
        ),
        name='pipeline example'
    )
]
report = analyzer.ml_regression(
    models=models,   # 5-fold cross validation for hyperparameter search
    y_var="target",
    X_vars=["age", "bmi", "bp", "s1", "s2"],
    outer_cv=5      # 5-fold cross validation for model evaluation
)
print(report.fit_statistics("train"))
print(report.cv_fit_statistics())
print(report.fit_statistics("test"))
reshow(report.model_report("TreeEnsembleR(adaboost)").test_report().plot_obs_vs_pred())
```

### Example UI + AI agent usage

TBD



### Demos

To learn more about TabularMagic functionality, check out the demos available in
the `./demo` subdirectory. 



## Development notes

Under active development. 









