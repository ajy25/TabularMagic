import pytest
import pandas as pd
import numpy as np
import pathlib
import sys

parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))


import tabularmagic as tm
from tabularmagic._src.data.preprocessing import (
    MinMaxSingleVar,
    StandardizeSingleVar,
    LogTransformSingleVar,
    Log1PTransformSingleVar,
)


@pytest.fixture
def setup_data():
    y = np.arange(20)
    y[0] = 1
    y = y * 100

    df_simple = pd.DataFrame({"y": y, "x1": np.arange(20), "x2": np.arange(20)[::-1]})

    return {"df_simple": df_simple}


def test_ols_scaling_simple(setup_data):
    """Tests minmax and standard scaling on a simple dataset."""

    df_simple = setup_data["df_simple"]

    analyzer = tm.Analyzer(df_simple, test_size=0.2, verbose=False)

    # quick minmax test
    rmse_unscaled = (
        analyzer.lm(target="y").metrics("test").loc["rmse", "Ordinary Least Squares"]
    )

    analyzer.scale(strategy="minmax")

    report = analyzer.lm(target="y")
    rmse_scaled = report.metrics("test").loc["rmse", "Ordinary Least Squares"]

    assert isinstance(report.model()._dataemitter.y_scaler(), MinMaxSingleVar)

    assert pytest.approx(rmse_unscaled) == pytest.approx(rmse_scaled)

    # once we reset the data, we should remove the y scaler
    analyzer.load_data_checkpoint()

    report = analyzer.lm(target="y")
    rmse_unscaled_2 = report.metrics("test").loc["rmse", "Ordinary Least Squares"]

    assert report.model()._dataemitter.y_scaler() is None

    assert pytest.approx(rmse_unscaled) == pytest.approx(rmse_unscaled_2)

    # quick standardize test
    analyzer.scale(strategy="standardize")
    report = analyzer.lm(target="y")
    rmse_scaled = report.metrics("test").loc["rmse", "Ordinary Least Squares"]
    assert isinstance(report.model()._dataemitter.y_scaler(), StandardizeSingleVar)
    assert pytest.approx(rmse_unscaled) == pytest.approx(rmse_scaled)

    # quick dual scale test
    analyzer.load_data_checkpoint()
    analyzer.scale(include_vars=["y"], strategy="standardize")
    analyzer.scale(include_vars=["x1"], strategy="minmax")
    report = analyzer.lm(target="y")
    rmse_scaled = report.metrics("test").loc["rmse", "Ordinary Least Squares"]
    assert isinstance(report.model()._dataemitter.y_scaler(), StandardizeSingleVar)
    assert pytest.approx(rmse_unscaled) == pytest.approx(rmse_scaled)


def test_ols_scaling_log(setup_data):
    """Tests log scaling on a simple dataset."""

    df_simple = setup_data["df_simple"]

    analyzer = tm.Analyzer(df_simple, test_size=0.2, verbose=False)

    report = analyzer.lm(target="y")
    assert report.model()._dataemitter.y_scaler() is None

    analyzer.scale(include_vars=["y"], strategy="log")

    report = analyzer.lm(target="y")
    assert isinstance(report.model()._dataemitter.y_scaler(), LogTransformSingleVar)

    analyzer.load_data_checkpoint()
    analyzer.scale(include_vars=["y"], strategy="log1p")
    report = analyzer.lm(target="y")
    assert isinstance(report.model()._dataemitter.y_scaler(), Log1PTransformSingleVar)


def test_ols_scaling_with_formula(setup_data):
    """Tests minmax and standard scaling on a simple dataset."""

    df_simple = setup_data["df_simple"]

    analyzer = tm.Analyzer(df_simple, test_size=0.2, verbose=False)

    # quick minmax test
    report = analyzer.lm(formula="log(y) ~ x1 + x2")
    assert isinstance(report.model()._dataemitter.y_scaler(), LogTransformSingleVar)
    rmse_1 = report.metrics("test").loc["rmse", "Ordinary Least Squares"]

    analyzer.scale(include_vars=["y"], strategy="log")
    report = analyzer.lm(formula="y ~ x1 + x2")
    assert isinstance(report.model()._dataemitter.y_scaler(), LogTransformSingleVar)
    rmse_2 = report.metrics("test").loc["rmse", "Ordinary Least Squares"]

    assert pytest.approx(rmse_1) == pytest.approx(rmse_2)


def test_ml_scaling_simple(setup_data):
    """Tests minmax and standard scaling on a simple dataset."""

    df_simple = setup_data["df_simple"]

    analyzer = tm.Analyzer(df_simple, test_size=0.2, verbose=False)

    # quick minmax test
    report = analyzer.regress(models=[tm.ml.LinearR(name="ols")], target="y")
    rmse_unscaled = report.metrics("test").loc["rmse", "ols"]
    assert report.model("ols")._dataemitter.y_scaler() is None

    analyzer.scale(strategy="minmax")
    report = analyzer.regress(models=[tm.ml.LinearR(name="ols")], target="y")
    assert isinstance(report.model("ols")._dataemitter.y_scaler(), MinMaxSingleVar)
    rmse_scaled = report.metrics("test").loc["rmse", "ols"]

    assert pytest.approx(rmse_unscaled) == pytest.approx(rmse_scaled)

    # once we reset the data, we should remove the y scaler
    analyzer.load_data_checkpoint()
    rmse_unscaled_2 = (
        analyzer.regress(models=[tm.ml.LinearR(name="ols")], target="y")
        .metrics("test")
        .loc["rmse", "ols"]
    )

    assert pytest.approx(rmse_unscaled) == pytest.approx(rmse_unscaled_2)

    # quick standardize test
    analyzer.scale(strategy="standardize")
    rmse_scaled = (
        analyzer.regress(models=[tm.ml.LinearR(name="ols")], target="y")
        .metrics("test")
        .loc["rmse", "ols"]
    )
    assert pytest.approx(rmse_unscaled) == pytest.approx(rmse_scaled)

    analyzer.load_data_checkpoint()
    analyzer.scale(include_vars=["y"], strategy="log")
    report = analyzer.regress(models=[tm.ml.LinearR(name="ols")], target="y")
    assert isinstance(
        report.model("ols")._dataemitter.y_scaler(), LogTransformSingleVar
    )

    analyzer.load_data_checkpoint()
    analyzer.scale(include_vars=["y"], strategy="log1p")
    report = analyzer.regress(
        models=[tm.ml.LinearR(name="ols"), tm.ml.TreesR(name="tree", n_trials=3)],
        target="y",
    )
    assert isinstance(
        report.model("ols")._dataemitter.y_scaler(), Log1PTransformSingleVar
    )
    assert isinstance(
        report.model("tree")._dataemitter.y_scaler(), Log1PTransformSingleVar
    )
