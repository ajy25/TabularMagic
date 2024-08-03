import pytest
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pathlib
import sys

parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from tabularmagic._src.data.datahandler import (
    DataEmitter,
    DataHandler,
    PreprocessStepTracer,
)


@pytest.fixture
def setup_data():
    df_simple = pd.DataFrame(
        {
            "binary_var": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            "categorical_var": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
            "numeric_var": [5.2, 3.7, 8.1, 2.5, 6.9, 4.3, 7.8, 1.1, 3.4, 5.7],
        }
    )
    df_simple_train, df_simple_test = train_test_split(
        df_simple, test_size=0.2, random_state=42
    )
    iris = datasets.load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    target = pd.Series(iris.target, name="target")
    df_iris = pd.concat([data, target], axis=1)

    df_iris_train, df_iris_test = train_test_split(
        df_iris, test_size=0.2, random_state=42
    )

    df_house = pd.read_csv(
        parent_dir / "demo" / "regression" / "house_price_data" / "data.csv"
    )
    df_house_train, df_house_test = train_test_split(
        df_house, test_size=0.2, random_state=42
    )

    return {
        "df_simple_train": df_simple_train,
        "df_simple_test": df_simple_test,
        "df_iris_train": df_iris_train,
        "df_iris_test": df_iris_test,
        "df_house_train": df_house_train,
        "df_house_test": df_house_test,
    }


def test_basic_init(setup_data):
    """Test basic initialization DataEmitter functionality."""
    df_simple_train = setup_data["df_simple_train"]
    df_simple_test = setup_data["df_simple_test"]

    dh = DataHandler(df_simple_train, df_simple_test, verbose=False)
    de = DataEmitter(
        df_simple_train,
        df_simple_test,
        "binary_var",
        ["categorical_var", "numeric_var"],
        PreprocessStepTracer(),
    )
    dh_emitter = dh.train_test_emitter(
        "binary_var", ["categorical_var", "numeric_var"]
    )
    assert dh_emitter._working_df_test.shape == de._working_df_test.shape
    assert dh_emitter._working_df_train.shape == de._working_df_train.shape
    dh.drop_vars(["binary_var"])
    assert dh._working_df_train.shape != de._working_df_train.shape


def test_force_categorical(setup_data):
    """Test force categorical encoding of numeric or binary variables."""
    df_simple_train = setup_data["df_simple_train"]
    df_simple_test = setup_data["df_simple_test"]

    dh = DataHandler(df_simple_train, df_simple_test, verbose=False)
    de = DataEmitter(
        df_simple_train,
        df_simple_test,
        "binary_var",
        ["categorical_var", "numeric_var"],
        PreprocessStepTracer(),
    )
    dh.force_categorical(["binary_var"])
    dh_emitter = dh.train_test_emitter(
        "binary_var", ["categorical_var", "numeric_var"]
    )
    assert dh_emitter._working_df_train["binary_var"].dtype == "object"
    assert dh_emitter._working_df_test.shape == de._working_df_test.shape
    assert dh_emitter._working_df_train.shape == de._working_df_train.shape


# Continue converting the rest of the test methods in a similar manner...


def test_force_numeric(setup_data):
    """Test force numeric encoding of categorical variables."""
    df_simple_train = setup_data["df_simple_train"]
    df_simple_test = setup_data["df_simple_test"]

    dh = DataHandler(df_simple_train, df_simple_test, verbose=False)
    de = DataEmitter(
        df_simple_train,
        df_simple_test,
        "binary_var",
        ["categorical_var", "numeric_var"],
        PreprocessStepTracer(),
    )
    dh.force_numeric(["binary_var"])
    dh_emitter = dh.train_test_emitter(
        "binary_var", ["categorical_var", "numeric_var"]
    )
    assert dh_emitter._working_df_train["binary_var"].dtype == "float64"
    assert dh_emitter._working_df_test.shape == de._working_df_test.shape
    assert dh_emitter._working_df_train.shape == de._working_df_train.shape


def test_force_binary(setup_data):
    """Test force binary encoding of numeric or categorical variables."""
    df_simple_train = setup_data["df_simple_train"]
    df_simple_test = setup_data["df_simple_test"]

    dh = DataHandler(df_simple_train, df_simple_test, verbose=False)
    de = DataEmitter(
        df_simple_train,
        df_simple_test,
        "binary_var",
        ["categorical_var", "numeric_var"],
        PreprocessStepTracer(),
    )
    dh.force_binary(["numeric_var"], rename=True)
    dh_emitter = dh.train_test_emitter(
        "binary_var", ["categorical_var", "numeric_var"]
    )
    assert (
        dh_emitter._working_df_train["numeric_var"].dtype
        == de._working_df_train["numeric_var"].dtype
    )
    assert dh_emitter._working_df_test.shape == de._working_df_test.shape
    assert dh_emitter._working_df_train.shape == de._working_df_train.shape

    dh = DataHandler(df_simple_train, df_simple_test, verbose=False)
    de = DataEmitter(
        df_simple_train,
        df_simple_test,
        "binary_var",
        ["categorical_var", "numeric_var"],
        PreprocessStepTracer(),
    )
    de._force_binary(
        ["categorical_var"], pos_labels=["A"], ignore_multiclass=True, rename=True
    )
    dh.force_binary(
        ["categorical_var"], pos_labels=["A"], ignore_multiclass=True, rename=True
    )
    dh_emitter = dh.train_test_emitter(
        "binary_var", ["A_yn(categorical_var)", "numeric_var"]
    )
    assert (
        dh_emitter._working_df_train["A_yn(categorical_var)"].dtype
        == de._working_df_train["A_yn(categorical_var)"].dtype
    )
    assert dh_emitter._working_df_test.shape == de._working_df_test.shape
    assert dh_emitter._working_df_train.shape == de._working_df_train.shape

    dh.force_binary(
        ["numeric_var"], pos_labels=[5.2], ignore_multiclass=True, rename=True
    )
    dh_emitter = dh.train_test_emitter(
        "binary_var", ["A_yn(categorical_var)", "5.2_yn(numeric_var)"]
    )
    assert dh._working_df_train["5.2_yn(numeric_var)"].dtype == "int64"
    assert dh_emitter._working_df_train.shape == dh._working_df_train.shape


def test_onehot(setup_data):
    """Test onehot encoding of categorical variables."""
    df_simple_train = setup_data["df_simple_train"]
    df_simple_test = setup_data["df_simple_test"]

    dh = DataHandler(df_simple_train, df_simple_test, verbose=False)
    dh.onehot(["categorical_var"], dropfirst=False)
    assert all(
        [
            col in dh._working_df_train.columns
            for col in [
                "A_yn(categorical_var)",
                "B_yn(categorical_var)",
                "C_yn(categorical_var)",
            ]
        ]
    )
    assert all(
        [
            col in dh._working_df_test.columns
            for col in [
                "A_yn(categorical_var)",
                "B_yn(categorical_var)",
                "C_yn(categorical_var)",
            ]
        ]
    )

    dh_emitter = dh.train_test_emitter(
        "binary_var",
        [
            "A_yn(categorical_var)",
            "B_yn(categorical_var)",
            "C_yn(categorical_var)",
            "numeric_var",
        ],
    )
    assert all(
        [
            col in dh_emitter._working_df_test.columns
            for col in [
                "A_yn(categorical_var)",
                "B_yn(categorical_var)",
                "C_yn(categorical_var)",
            ]
        ]
    )
    assert all(
        [
            col in dh_emitter._working_df_train.columns
            for col in [
                "A_yn(categorical_var)",
                "B_yn(categorical_var)",
                "C_yn(categorical_var)",
            ]
        ]
    )
    assert dh_emitter._working_df_test.shape == dh._working_df_test.shape
    assert dh_emitter._working_df_train.shape == dh._working_df_train.shape

    dh = DataHandler(df_simple_train, df_simple_test, verbose=False)
    dh.onehot(["categorical_var", "binary_var"], dropfirst=True)
    assert all(
        [
            col in dh._working_df_train.columns
            for col in ["B_yn(categorical_var)", "C_yn(categorical_var)"]
        ]
    )
    assert all(
        [
            col in dh._working_df_test.columns
            for col in ["B_yn(categorical_var)", "C_yn(categorical_var)"]
        ]
    )
    assert "A_yn(categorical_var)" not in dh._working_df_train.columns

    dh_emitter = dh.train_test_emitter(
        "numeric_var",
        ["B_yn(categorical_var)", "C_yn(categorical_var)", "1_yn(binary_var)"],
    )
    assert all(
        [
            col in dh_emitter._working_df_test.columns
            for col in ["B_yn(categorical_var)", "C_yn(categorical_var)"]
        ]
    )
    assert all(
        [
            col in dh_emitter._working_df_train.columns
            for col in ["B_yn(categorical_var)", "C_yn(categorical_var)"]
        ]
    )
    assert dh_emitter._working_df_test.shape == dh._working_df_test.shape
    assert dh_emitter._working_df_train.shape == dh._working_df_train.shape


def test_multiple(setup_data):
    """Test multiple preprocessing steps."""
    df_iris_train = setup_data["df_iris_train"]
    df_iris_test = setup_data["df_iris_test"]

    dh = DataHandler(df_iris_train, df_iris_test, verbose=False)
    dh.force_categorical(["target"])
    dh.scale(["sepallength(cm)"], strategy="log")
    dh.drop_vars(["sepalwidth(cm)"])
    dh_emitter = dh.train_test_emitter(
        "target", ["sepallength(cm)", "petallength(cm)", "petalwidth(cm)"]
    )
    assert all(
        col1 == col2
        for col1, col2 in zip(
            dh_emitter._working_df_train.columns, dh._working_df_train.columns
        )
    )
    assert dh_emitter._working_df_train.shape == dh._working_df_train.shape
    assert all(
        idx1 == idx2
        for idx1, idx2 in zip(
            dh_emitter._working_df_train.index, dh._working_df_train.index
        )
    )


def test_scale(setup_data):
    """Test scaling of numeric variables."""
    df_iris_train = setup_data["df_iris_train"]
    df_iris_test = setup_data["df_iris_test"]

    dh = DataHandler(df_iris_train, df_iris_test, verbose=False)
    dh.scale(strategy="minmax")
    assert dh._working_df_train["sepallength(cm)"].min() == pytest.approx(0)
    assert dh._working_df_train["sepallength(cm)"].max() == pytest.approx(1)
    assert dh._working_df_train["petallength(cm)"].min() == pytest.approx(0)
    assert dh._working_df_train["petallength(cm)"].max() == pytest.approx(1)

    dh_emitter = dh.train_test_emitter(
        "target", ["sepallength(cm)", "petallength(cm)", "petalwidth(cm)"]
    )
    assert dh_emitter._working_df_train["sepallength(cm)"].min() == pytest.approx(0)
    assert dh_emitter._working_df_train["sepallength(cm)"].max() == pytest.approx(1)
    assert dh_emitter._working_df_train["petallength(cm)"].min() == pytest.approx(0)
    assert dh_emitter._working_df_train["petallength(cm)"].max() == pytest.approx(1)

    dh = DataHandler(df_iris_train, df_iris_test, verbose=False)
    dh.scale(strategy="standardize")
    assert dh._working_df_train["sepallength(cm)"].mean() == pytest.approx(0, abs=0.02)
    assert dh._working_df_train["sepallength(cm)"].std() == pytest.approx(1, abs=0.02)
    assert dh._working_df_train["petallength(cm)"].mean() == pytest.approx(0, abs=0.02)
    assert dh._working_df_train["petallength(cm)"].std() == pytest.approx(1, abs=0.02)

    dh_emitter = dh.train_test_emitter(
        "target", ["sepallength(cm)", "petallength(cm)", "petalwidth(cm)"]
    )
    assert dh_emitter._working_df_train["sepallength(cm)"].mean() == pytest.approx(
        0, abs=0.02
    )
    assert dh_emitter._working_df_train["sepallength(cm)"].std() == pytest.approx(
        1, abs=0.02
    )

    dh_emitters = dh.kfold_emitters(
        "target", ["sepallength(cm)", "petallength(cm)", "petalwidth(cm)"]
    )
    for emitter in dh_emitters:
        assert emitter._working_df_train["sepallength(cm)"].mean() == pytest.approx(
            0, abs=0.02
        )
        assert emitter._working_df_train["sepallength(cm)"].std() == pytest.approx(
            1, abs=0.1
        )


def test_drop_vars(setup_data):
    """Test dropping variables."""
    df_iris_train = setup_data["df_iris_train"]
    df_iris_test = setup_data["df_iris_test"]

    dh = DataHandler(df_iris_train, df_iris_test, verbose=False)
    dh.drop_vars(["sepallength(cm)"])
    assert "sepallength(cm)" not in dh._working_df_train.columns
    assert "sepallength(cm)" not in dh._working_df_test.columns

    dh_emitter = dh.train_test_emitter("target", ["petallength(cm)", "petalwidth(cm)"])
    assert "sepallength(cm)" not in dh_emitter._working_df_train.columns
    assert "sepallength(cm)" not in dh_emitter._working_df_test.columns


def test_impute(setup_data):
    """Test imputing missing values."""
    df_house_train = setup_data["df_house_train"]
    df_house_test = setup_data["df_house_test"]

    dh = DataHandler(df_house_train, df_house_test, verbose=False)
    dh.impute(numeric_strategy="mean", categorical_strategy="most_frequent")
    assert all(
        not dh._working_df_train[col].isnull().any()
        for col in dh._working_df_train.columns
    )
    assert all(
        not dh._working_df_test[col].isnull().any()
        for col in dh._working_df_test.columns
    )

    dh_emitter = dh.train_test_emitter(
        "SalePrice", ["LotFrontage", "LotArea", "OverallQual", "OverallCond"]
    )
    assert all(
        not dh_emitter._working_df_train[col].isnull().any()
        for col in dh_emitter._working_df_train.columns
    )
    assert all(
        not dh_emitter._working_df_test[col].isnull().any()
        for col in dh_emitter._working_df_test.columns
    )

    dh_emitters = dh.kfold_emitters(
        "SalePrice", ["LotFrontage", "LotArea", "OverallQual", "OverallCond"]
    )
    for emitter in dh_emitters:
        assert all(
            not emitter._working_df_train[col].isnull().any()
            for col in emitter._working_df_train.columns
        )
        assert all(
            not emitter._working_df_test[col].isnull().any()
            for col in emitter._working_df_test.columns
        )

    assert df_house_train["LotFrontage"].isnull().any()
    assert df_house_train["MasVnrArea"].isnull().any()
    dh = DataHandler(df_house_train, df_house_test, verbose=False)
    dh.impute(
        include_vars=["LotFrontage", "MasVnrArea", "BsmtFinSF1"],
        exclude_vars=["LotFrontage"],
    )
    assert dh._working_df_train["LotFrontage"].isnull().any()
    assert not dh._working_df_train["MasVnrArea"].isnull().any()
    assert df_house_train["MasVnrArea"].isnull().any()


def test_kfold_basic_init(setup_data):
    """Test kfold cross validation basic functionality."""
    df_iris_train = setup_data["df_iris_train"]
    df_iris_test = setup_data["df_iris_test"]

    dh = DataHandler(df_iris_train, df_iris_test, verbose=False)
    dh.force_categorical(["target"])
    dh.scale(["sepallength(cm)"], strategy="log")
    dh.drop_vars(["sepalwidth(cm)"])
    emitters = dh.kfold_emitters(
        "target", ["sepallength(cm)", "petallength(cm)", "petalwidth(cm)"]
    )
    idxs = []
    for emitter in emitters:
        for col1, col2 in zip(
            emitter._working_df_train.columns, dh._working_df_train.columns
        ):
            assert col1 == col2
        idxs.append(emitter._working_df_train.index.to_list())
    idxs = np.concatenate(idxs)
    for idx in idxs:
        assert idx in dh._working_df_train.index

    dh = DataHandler(df_iris_train, df_iris_test, verbose=False)
    dh.force_categorical(["target"])
    dh.onehot(["target"])
    dh.scale(["sepallength(cm)"], strategy="minmax")
    dh.drop_vars(["sepalwidth(cm)"])
    emitters = dh.kfold_emitters(
        "sepallength(cm)",
        ["1_yn(target)", "2_yn(target)", "petallength(cm)", "petalwidth(cm)"],
        n_folds=5,
        shuffle=True,
        random_state=42,
    )
    idxs = []
    for emitter in emitters:
        for col1, col2 in zip(
            emitter._working_df_train.columns, dh._working_df_train.columns
        ):
            assert col1 == col2
    idxs.append(emitter._working_df_train.index.to_list())
    idxs = np.concatenate(idxs)
    for idx in idxs:
        assert idx in dh._working_df_train.index
    assert emitters[0].y_scaler().min == 4.3
    assert emitters[2].y_scaler().min != 4.3


def test_emitter_feature_selection(setup_data):
    df_iris_train = setup_data["df_iris_train"]
    df_iris_test = setup_data["df_iris_test"]

    dh = DataHandler(df_iris_train, df_iris_test, verbose=False)
    emitter = dh.train_test_emitter(
        "target", ["sepallength(cm)", "petallength(cm)", "petalwidth(cm)"]
    )
    emitter.select_predictors(["sepallength(cm)", "petallength(cm)"])
    assert np.allclose(
        emitter.emit_train_Xy()[0].to_numpy(),
        df_iris_train[["sepallength(cm)", "petallength(cm)"]].to_numpy(),
    )
    assert np.allclose(
        emitter.emit_test_Xy()[0].to_numpy(),
        df_iris_test[["sepallength(cm)", "petallength(cm)"]].to_numpy(),
    )
    assert np.allclose(
        emitter.emit_train_Xy()[1].to_numpy(), df_iris_train["target"].to_numpy()
    )
    assert np.allclose(
        emitter.emit_test_Xy()[1].to_numpy(), df_iris_test["target"].to_numpy()
    )

    df_house_train = setup_data["df_house_train"]
    df_house_test = setup_data["df_house_test"]

    dh = DataHandler(df_house_train, df_house_test, verbose=False)
    all_vars = ["SalePrice", "LotFrontage", "LotArea", "OverallQual", "LotShape"]
    emitter = dh.train_test_emitter("SalePrice", all_vars[1:])
    emitter.select_predictors(["OverallQual", "LotArea"])
    assert np.allclose(
        emitter.emit_train_Xy()[0].to_numpy(),
        df_house_train[all_vars].dropna()[["OverallQual", "LotArea"]].to_numpy(),
    )
    assert np.allclose(
        emitter.emit_test_Xy()[0].to_numpy(),
        df_house_test[all_vars].dropna()[["OverallQual", "LotArea"]].to_numpy(),
    )
    assert np.allclose(
        emitter.emit_train_Xy()[1].to_numpy(),
        df_house_train[all_vars].dropna()["SalePrice"].to_numpy(),
    )
    assert np.allclose(
        emitter.emit_test_Xy()[1].to_numpy(),
        df_house_test[all_vars].dropna()["SalePrice"].to_numpy(),
    )


def test_emitter_feature_selection_transform(setup_data):
    df_house_train = setup_data["df_house_train"]
    df_house_test = setup_data["df_house_test"]

    dh = DataHandler(df_house_train, df_house_test, verbose=False)


def test_datahandler_scale_(setup_data):
    """Test DataEmitter with basic pipeline generation capabilities"""
    train_data = setup_data["df_house_train"]
    test_data = setup_data["df_house_test"]
    dh = DataHandler(
        train_data,
        test_data,
        verbose=False
    )

    dh.scale(
        include_vars=[
            "GrLivArea", "YearBuilt", "OverallQual", "LotArea", "SalePrice"
        ],
        strategy="standardize"
    )

    assert 'MSZoning' in dh.df_train().columns
    dh.onehot(['MSZoning'])

