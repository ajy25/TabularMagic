import sys
import pathlib
import numpy as np
import pandas as pd
import pytest

parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))


import tablemage as tm


@pytest.fixture
def setup_data():
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "x1": np.random.randn(100) * 0.5 + 10,  # Continuous variable
            "x2": np.random.randn(100) * 3,  # Continuous variable (noise)
            "cat1": np.random.choice(["A", "B", "C"], 100),  # Categorical variable
            "cat2": np.random.choice(
                ["W", "X", "Y", "Z"], 100
            ),  # Categorical variable (noise)
            "x3": np.random.randn(100) * 3,  # Continuous variable (noise)
        }
    )

    # Dependent variable
    y = (
        3 * X["x1"]
        - 5 * (X["cat1"] == "A")
        + 10 * (X["cat1"] == "B")
        + np.random.randn(100)
    )

    df_ols = X
    df_ols["y"] = y

    np.random.seed(42)
    X = pd.DataFrame(
        {
            "x1": np.random.randn(100) * 0.5 + 10,  # Continuous variable
            "x2": np.random.randn(100) * 3,  # Continuous variable (noise)
            "cat1": np.random.choice(["A", "B", "C"], 100),  # Categorical variable
            "cat2": np.random.choice(
                ["W", "X", "Y", "Z"], 100
            ),  # Categorical variable (noise)
            "x3": np.random.randn(100) * 3,  # Continuous variable (noise)
        }
    )

    # Create a binary target variable
    logit = (
        3 * X["x1"]
        - 5 * (X["cat1"] == "A")
        + 10 * (X["cat1"] == "B")
        + np.random.randn(100) * 3
        - 3
    )
    y = (logit > logit.median()).astype(int)

    df_logistic = X
    df_logistic["y"] = y
    df_logistic["y"] = df_logistic["y"].astype(str)

    return {
        "df_ols": df_ols,
        "df_logistic": df_logistic,
    }


def test_output_errorless(setup_data):
    """Simply need to run without errors"""
    df = setup_data["df_ols"]

    analyzer = tm.Analyzer(df=df)
    lmreport = analyzer.ols(
        target="y",
        predictors=["x1", "x2", "cat1", "cat2", "x3"],
    )

    lmreport.coefs()
    lmreport.statsmodels_summary()
    lmreport.metrics("train")
    lmreport.metrics("test")
    lmreport.plot_diagnostics("train")
    lmreport.plot_diagnostics("test")
    lmreport.plot_residuals_vs_var("x1", "train")
