# TabularMagic

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/tabularmagic/badge/?version=latest)](https://tabularmagic.readthedocs.io/en/latest/?badge=latest)
![Tests Passing](https://github.com/ajy25/TabularMagic/actions/workflows/test.yml/badge.svg)


TabularMagic is a Python package for rapid exploratory statistical analysis and low-code machine learning modeling on wide format tabular data. TabularMagic can help you quickly explore datasets, easily conduct regression analyses, and effortlessly compute performance metrics for your favorite machine learning models.


## Installation and dependencies

TabularMagic can be installed from source.

To install TabularMagic:
```
git clone https://github.com/ajy25/TabularMagic.git
cd TabularMagic
pip install .
```

To uninstall TabularMagic:
```
pip uninstall tabularmagic
```

TabularMagic is built with the standard Python data science stack. That is, TabularMagic is really just a fancy wrapper for scikit-learn and statsmodels. For additional notes regarding dependencies, check out `./dev_notes/dependencies.md`. TabularMagic requires Python version 3.10 or later.


## Quick start

You'll probably use TabularMagic for ML model benchmarking. Here's how to do it.

```python
import tabularmagic as tm
import pandas as pd
import joblib

df = ...

# initialize an Analyzer object
analyzer = tm.Analyzer(df, test_size=0.2)

# preprocess data
analyzer.drop_highly_missing_vars().dropna(['y']).impute().scale()

# conduct a regression model benchmarking exercise
reg_report = analyzer.regress(
    models=[
        tm.ml.LinearR('l2'),
        tm.ml.TreesR('random_forest'),
        tm.ml.TreesR('xgboost'),
    ],
    target='y',
    feature_selectors=[
        tm.fs.BorutaFSR()
    ]
)
print(reg_report.metrics('test'))

# predict on new data
new_df = ...
y_pred = reg_report.model('LinearR(l2)').predict(new_df)

# save model
joblib.dump(reg_report.model('LinearR(l2)'), 'l2_pipeline.joblib')
```

Check out the `./demo` directory for detailed examples and discussion of other functionality.


## Notes

TabularMagic is under active development. We intend to release a test version to Test PyPI soon.

### Motivation: low-code data science and ML modeling for research purposes

Though numerous open-source automatic/low-code machine learning packages have emerged to streamline model selection and deployment, packages tailored specifically for research on tabular datasets remain scarce.

TabularMagic provides a straightforward Python API that exponentially accelerates machine learning model benchmarking by seamlessly connecting the data exploration and processing steps to the modeling steps. TabularMagic offers the following:
1. **Preprocess-as-you-explore functionality.** TabularMagic remembers each feature transformation you make and automatically preprocesses your train, validation, and test datasets when you fit and evaluate models down the line. 
2. **Automatic hyperparameter optimization and feature selection.** TabularMagic can automatically select features and identify optimal hyperparameters for you. All TabularMagic ML models come with preset hyperparameter search methods. 
3. **Flexibility.** Though TabularMagic provides many out-of-the-box models with default hyperparameter search spaces, it also supports custom estimators and pipelines. Any scikit-learn `BaseEstimator`/`Pipeline`-like object with fit and predict methods can be used. 
4. **Linear regression.** TabularMagic contains numerous methods to support statsmodels' classical linear statistical models, including diagnostic plots, stepwise feature selection, and statistical tests, enabling you to seamlessly switch between linear statistical modeling and ML modeling.
5. **LLM support.**  (coming soon!) TabularMagic comes equipped with LangChain LLM agents and tools that allow you to chat with your data. 

See more in `./dev_notes/notes.md`.







