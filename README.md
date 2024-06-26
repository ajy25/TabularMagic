# TabularMagic

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

TabularMagic is a Python package for rapid exploratory statistical and machine learning modeling of wide format tabular data. TabularMagic empowers users to quickly explore new datasets, conduct regression analyses with ease, and effortlessly compute baseline performance metrics across a wide range of popular machine learning models. TabularMagic excels in handling datasets with fewer than 10,000 examples. 

Under active development.


### Why does TabularMagic exist?

Though numerous auto-ML solutions have emerged to streamline data science workflows at an enterprise scale, low-code data science packages tailored for small tabular datasets remain scarce. TabularMagic strives to fill this void, offering a straightforward Python interface for common data science routines. This package relieves users from the tedious tasks often associated with such projects – maintaining separate train and test data, one-hot encoding and scaling features, and proper cross-validation benchmarking of various machine learning models, many of which require hyperparameter tuning.


## Getting started

### Installation and dependencies

TabularMagic can be installed via pip. The Python scripts below handle 
package setup and pip installation. 

To install TabularMagic: 
```
git clone https://github.com/ajy25/TabularMagic.git
cd tabularmagic
python tmbuild.py install
```

To uninstall TabularMagic:
```
python tmbuild.py uninstall
```

TabularMagic is built on top of the standard Python data science stack (scikit-learn, statsmodels, pandas, NumPy, Matplotlib). 
A full list of dependencies is available in ```./requirements.txt```.


### Example usage

We can build an Analyzer object on a given dataset.
```
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
```
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

TabularMagic makes regression analysis easy (though admittedly not by much).
```
lm_report = analyzer.lm(
    formula="target ~ age + bmi"
)
lm_report.statsmodels_summary()
lm_report.train_report().set_outlier_threshold(2).plot_diagnostics(
    show_outliers=True)
```

TabularMagic makes machine learning model benchmarking easy. Nested k-fold cross validation handles hyperparameter selection and model evaluation on training data. The selected models are evaluated on the withheld testing data as well. Note that nested cross validation is computationally expensive and could take some time to run; to disable nested cross validation, simply set `outer_cv = None`.
```
models =[
    tm.ml.LinearR(),
    tm.ml.LinearR("l1"),
    tm.ml.LinearR("l2"),
    tm.ml.TreeEnsembleR("random_forest", n_jobs=-1),
    tm.ml.TreeEnsembleR("adaboost", n_jobs=-1),
    tm.ml.SVMR("rbf", n_jobs=-1)
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



### Demos

To learn more about TabularMagic functionality, check out the demos available in
the `./demo` subdirectory. 



## Development notes











