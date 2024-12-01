# TableMage

![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Tests Passing](https://github.com/ajy25/TableMage/actions/workflows/test.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/TableMage/badge/?version=latest)](https://TableMage.readthedocs.io/en/latest/?badge=latest)



TableMage is a Python package for low/no-code data science on wide format tables.
TableMage can help you quickly explore datasets, 
easily conduct statistical and regression analyses, 
and effortlessly compute performance metrics for your favorite machine learning models.


## Installation and dependencies

TableMage can be installed from source.

To install TableMage:
```
git clone https://github.com/ajy25/TableMage.git
cd TableMage
pip install .
```

To uninstall TableMage:
```
pip uninstall TableMage
```

TableMage is built with the standard Python data science stack.
That is, TableMage is really just a fancy wrapper for scikit-learn, scipy.stats, and statsmodels. 
For additional notes regarding dependencies, check out `./dev_notes/dependencies.md`. 
TableMage requires Python version 3.10 or later.


## Quick start (low-code)

You'll probably use TableMage for ML model benchmarking. Here's how to do it.

```python
import tablemage as tm
import pandas as pd
import joblib

# load table (we'll assume 'y' is the numeric variable to predict)
df = ...

# initialize an Analyzer object
analyzer = tm.Analyzer(df, test_size=0.2)

# preprocess data
analyzer.dropna(['y']).impute().scale()

# train regressors (hyperparameter tuning is preset and automatic)
reg_report = analyzer.regress(
    models=[
        tm.ml.LinearR('l2'),
        tm.ml.TreesR('random_forest'),
        tm.ml.TreesR('xgboost'),
    ],
    target='y',
    feature_selectors=[
        tm.fs.BorutaFSR()   # select features
    ]
)

# compare model performance
print(reg_report.metrics('test'))

# predict on new data
new_df = ...
y_pred = reg_report.model('LinearR(l2)').predict(new_df)

# save model as sklearn pipeline
joblib.dump(reg_report.model('LinearR(l2)'), 'l2_pipeline.joblib')
```

Check out the `./demo` directory for detailed examples and discussion of other functionality.




## Quick start (no-code)

```python
from tablemage.mage import App
App().run(debug=True)
```


## Notes

TableMage is under active development.

### Motivation: low-code data science and ML modeling for research purposes

Though numerous open-source automatic/low-code machine learning packages have emerged to streamline model selection and deployment, packages tailored specifically for research on tabular datasets remain scarce.

TableMage provides a straightforward Python API that exponentially accelerates machine learning model benchmarking by seamlessly connecting the data exploration and processing steps to the modeling steps. TableMage offers the following:
1. **Preprocess-as-you-explore functionality.** TableMage remembers each feature transformation you make and automatically preprocesses your train, validation, and test datasets when you fit and evaluate models down the line. 
2. **Automatic hyperparameter optimization and feature selection.** TableMage can automatically select features and identify optimal hyperparameters for you. All TableMage ML models come with preset hyperparameter search methods. 
3. **Flexibility.** Though TableMage provides many out-of-the-box models with default hyperparameter search spaces, it also supports custom estimators and pipelines. Any scikit-learn `BaseEstimator`/`Pipeline`-like object with fit and predict methods can be used. 
4. **Linear regression.** TableMage contains numerous methods to support statsmodels' classical linear statistical models, including diagnostic plots, stepwise feature selection, and statistical tests, enabling you to seamlessly switch between linear statistical modeling and ML modeling.
5. **LLM support.**  (coming soon!) TableMage comes equipped with LangChain LLM agents and tools that allow you to chat with your data. 

See more in `./.dev_notes/notes.md`.







