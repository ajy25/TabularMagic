# TabularMagic

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/tabularmagic/badge/?version=latest)](https://tabularmagic.readthedocs.io/en/latest/?badge=latest)


TabularMagic is a Python package for rapid exploratory statistical analysis and automatic machine learning modeling on wide format tabular data. TabularMagic can help you quickly explore datasets, easily conduct regression analyses, and effortlessly compute performance metrics for your favorite machine learning models.


## Installation and dependencies

TabularMagic can be installed via pip. The Python scripts below handle package setup and pip installation. 

To install TabularMagic: 
```
git clone https://github.com/ajy25/TabularMagic.git
cd TabularMagic
python tmbuild.py install
```

To uninstall TabularMagic:
```
python tmbuild.py uninstall
```

TabularMagic is built with the standard Python data science stack. For additional notes regarding dependencies, check out `./dev_notes/dependencies.md`. TabularMagic requires Python version 3.10 or later.

## Getting started

### Example Python API usage

Let's initialize an Analyzer.
```python
import pandas as pd
import matplotlib.pyplot as plt
import tabularmagic as tm
from sklearn.datasets import load_diabetes

# load the dataset
diabetes_data = load_diabetes()
df = pd.DataFrame(data=diabetes_data.data, columns=diabetes_data.feature_names)
df['target'] = diabetes_data.target

# initialize an Analyzer object
analyzer = tm.Analyzer(df, test_size=0.2, split_seed=42)
print(analyzer)
```

TabularMagic streamlines exploratory data analysis. 
*Note: all TabularMagic plotting methods close and return plt.Figures for easier use with IPython notebooks (e.g.* `display(fig)` *). When working outside of IPython notebooks (not recommended), returned plt.Figures must be reshown.*
```python
def reshow(fig: plt.Figure):
    new_fig = plt.figure(figsize=fig.get_size_inches())
    new_manager = new_fig.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    plt.show()

train_eda = analyzer.eda()
print(train_eda.numeric_stats())
reshow(train_eda.plot_distribution('target'))
reshow(train_eda.plot_numeric_pairs(['target', 'age', 'bmi', 'bp']))
```

TabularMagic streamlines regression analysis.
```python
lm_report = analyzer.lm(
    formula='target ~ age + bmi'
)
print(lm_report.statsmodels_summary())
reshow(lm_report.train_report().plot_diagnostics(show_outliers=True))
print(lm_report.metrics('test'))
```

TabularMagic streamlines machine learning model benchmarking. TabularMagic automatically tunes hyperparameters and selects features when benchmarking user-specified models. Note that cross validation is optional and computationally expensive; to enable cross validation, simply set `outer_cv` to the number of desired folds.
```python
models = [
    tm.ml.LinearR(),
    tm.ml.LinearR('l1'),
    tm.ml.LinearR('l2'),
    tm.ml.TreeEnsembleR('random_forest'),
    tm.ml.TreeEnsembleR('adaboost'),
    tm.ml.SVMR('rbf')
]
report = analyzer.regress(
    models=models, 
    target='target',
    predictors=['age', 'bmi', 'bp', 's1', 's2'],
    feature_selectors=[
        tm.fs.KBestSelectorR('f_regression', k=3),
        tm.fs.KBestSelectorR('mutual_info_regression', k=3),
    ],
    max_n_features=3, # voting feature selection, top 3 features
    inner_cv=5        # 5-fold cross validation for hyperparameter tuning
    outer_cv=5        # 5-fold cross validation for model evaluation
)
print(report.cv_metrics())      # cross validation metrics
print(report.metrics('train'))  # train metrics
print(report.metrics('test'))   # test metrics
reshow(report.plot_obs_vs_pred('TreeEnsembleR(adaboost)', 'test'))
```

### Example UI + AI agent usage

This feature is coming! We're working on it.



### Demos

To learn more about TabularMagic functionality, check out the demos available in
the `./demo` subdirectory. 



## Development notes

Under active development. We intend to push an initial releaase to Test PyPI soon.

### Motivation: auto ML for research, not production

Though numerous automatic machine learning solutions have emerged to streamline machine learning model selection and deployment, low-code libraries tailored for research on tabular datasets remain scarce.

TabularMagic provides a straightforward Python API that significantly accelerates machine learning model benchmarking by seemlessly connecting the data exploration and processing steps to the modeling steps. TabularMagic offers the following:
1. **Preprocess-as-you-explore functionality.** TabularMagic remembers each feature transformation you make and automatically preprocesses your train, validation, and test datasets when you fit and evaluate models down the line. 
2. **Automatic hyperparameter optimization and feature selection.** TabularMagic automatically selects features and identifies optimal hyperparameters for you.
3. **Flexibility.** Though TabularMagic provides many out-of-the-box models with default hyperparameter search spaces, it also supports custom estimators and pipelines. Any scikit-learn `BaseEstimator`/`Pipeline`-like object with fit and predict methods can be used. You'll need to specify the hyperparameter tuning strategy (e.g. `GridSearchCV`) yourself, however.
4. **LLM support.** TabularMagic comes equipped with LangChain LLM agents and tools that allow you to chat with your data.


### FAQs (why does TabularMagic exist?):

1. 
    Q: Why not just use sklearn pipelines? 

    A: scikit-learn is *the* Python machine learning modeling package; TabularMagic and many other solutions rely heavily on scikit-learn. Though sklearn pipelines allows for streamlined data preprocessing and ML modeling, they are by no means low-code and require a nontrivial amount of documentation reading and programming experience to use. 












