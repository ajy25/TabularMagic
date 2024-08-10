# TabularMagic

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/tabularmagic/badge/?version=latest)](https://tabularmagic.readthedocs.io/en/latest/?badge=latest)


TabularMagic is a Python package for rapid exploratory statistical analysis and low-code machine learning modeling on wide format tabular data. TabularMagic can help you quickly explore datasets, easily conduct regression analyses, and effortlessly compute performance metrics for your favorite machine learning models.


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


## Quick start

You'll probably be using TabularMagic for ML model benchmarking. Here's how to do it.

```python
import tabularmagic as tm
import pandas as pd

df = ...

# initialize an Analyzer object
analyzer = tm.Analyzer(df, test_size=0.2)

# scale and impute values (with default strategies, see docs for how to specify)
analyzer.scale().impute()

# conduct a regression model benchmarking exercise
reg_report = analyzer.regress(
    models=[
        tm.ml.LinearR('l2'),
        tm.ml.TreesR('random_forest'),
        tm.ml.TreesR('xgboost'),
    ],
    target='y',
    predictors=['x1', 'x2', 'x3']
)
print(reg_report.metrics('test'))
```

Check out the `./demo` directory for detailed examples and discussion of other functionality.


## Development notes

Under active development. We intend to push an initial releaase to Test PyPI soon.

### Motivation: low-code machine learning for research, not production

Though numerous open-source automatic/low-code machine learning packages have emerged to streamline model selection and deployment, packages tailored specifically for research on tabular datasets remain scarce.

TabularMagic provides a straightforward Python API that significantly accelerates machine learning model benchmarking by seemlessly connecting the data exploration and processing steps to the modeling steps. TabularMagic offers the following:
1. **Preprocess-as-you-explore functionality.** TabularMagic remembers each feature transformation you make and automatically preprocesses your train, validation, and test datasets when you fit and evaluate models down the line. 
2. **Automatic hyperparameter optimization and feature selection.** TabularMagic automatically selects features and identifies optimal hyperparameters for you.
3. **Flexibility.** Though TabularMagic provides many out-of-the-box models with default hyperparameter search spaces, it also supports custom estimators and pipelines. Any scikit-learn `BaseEstimator`/`Pipeline`-like object with fit and predict methods can be used. You'll need to specify the hyperparameter tuning strategy (e.g. `GridSearchCV`) yourself, though.
4. **LLM support.**  (coming soon!) TabularMagic comes equipped with LangChain LLM agents and tools that allow you to chat with your data.


### FAQs (why does TabularMagic exist?):

1. 
    Q: Why not just use scikit-learn for machine learning? 

    A: scikit-learn is *the* Python machine learning modeling package; TabularMagic and many other solutions rely heavily on scikit-learn. Though sklearn pipelines allows for streamlined data preprocessing and ML modeling, they are by no means low-code and require a nontrivial amount of documentation reading, programming experience, and machine learning knowledge to use. 


2. 
    Q: Are there si









