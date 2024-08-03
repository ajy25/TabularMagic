import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pathlib
import sys

parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))



import tabularmagic as tm



@pytest.fixture
def setup_data():
    df_house = pd.read_csv(
        parent_dir / "demo" / "regression" / "house_price_data" / "data.csv"
    )
    df_house_train, df_house_test = train_test_split(
        df_house, test_size=0.2, random_state=42
    )

    return {
        "df_house_train": df_house_train,
        "df_house_test": df_house_test,
    }



def test_standard_scaling(setup_data):
    """Tests basic feature scaling functionality"""

    df_house_train = setup_data["df_house_train"]
    df_house_test = setup_data["df_house_test"]

    analyzer = tm.Analyzer(
        df_house_train,
        df_house_test,
        verbose=False
    )

    scaled_vars = [
        "GrLivArea", "YearBuilt", "OverallQual", "LotArea", "SalePrice"
    ]

    analyzer.scale(
        include_vars=scaled_vars,
        strategy="standardize"
    )

    analyzer_working_df_train = analyzer.datahandler().df_train()[scaled_vars]

    scaler = StandardScaler()

    assert np.allclose(
        analyzer_working_df_train.to_numpy(), 
        scaler.fit_transform(
            df_house_train[scaled_vars].to_numpy()
        ),
        atol=1e-5
    )

    analyzer_working_df_test = analyzer.datahandler().df_test()[scaled_vars]

    assert np.allclose(
        analyzer_working_df_test.to_numpy(), 
        scaler.transform(
            df_house_test[scaled_vars].to_numpy()
        ),
        atol=1e-5
    )


def test_minmax_scaling(setup_data):
    """Tests basic feature scaling functionality"""

    df_house_train = setup_data["df_house_train"]
    df_house_test = setup_data["df_house_test"]

    analyzer = tm.Analyzer(
        df_house_train,
        df_house_test,
        verbose=False
    )

    scaled_vars = [
        "GrLivArea", "YearBuilt", "OverallQual", "LotArea", "SalePrice"
    ]

    analyzer.scale(
        include_vars=scaled_vars,
        strategy="minmax"
    )

    analyzer_working_df_train = analyzer.datahandler().df_train()[scaled_vars]

    scaler = MinMaxScaler()

    assert np.allclose(
        analyzer_working_df_train.to_numpy(), 
        scaler.fit_transform(
            df_house_train[scaled_vars].to_numpy()
        ),
        atol=1e-5
    )

    analyzer_working_df_test = analyzer.datahandler().df_test()[scaled_vars]

    assert np.allclose(
        analyzer_working_df_test.to_numpy(), 
        scaler.transform(
            df_house_test[scaled_vars].to_numpy()
        ),
        atol=1e-5
    )



