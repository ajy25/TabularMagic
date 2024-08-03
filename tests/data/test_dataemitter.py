import pytest
import pandas as pd
import numpy as np
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




def test_dataemitter_pipeline_basic(setup_data):
    """Test DataEmitter with basic pipeline generation capabilities"""
    train_data = setup_data["df_house_train"]
    test_data = setup_data["df_house_test"]
    dh = DataHandler(
        train_data,
        test_data,
        verbose=False
    )
    de = dh.train_test_emitter(
        y_var="SalePrice", 
        X_vars=["GrLivArea", "YearBuilt", "OverallQual", "LotArea", "LotShape"],
    )


    transformer = de.sklearn_preprocessing_transformer()

    test_data_subset = test_data[
        ["GrLivArea", "YearBuilt", "OverallQual", "LotArea", "LotShape", "SalePrice"]
    ]

    transformer_df = transformer.transform(test_data_subset)

    emitted_test_Xy = de.emit_test_Xy()
    emitted_df = emitted_test_Xy[0].join(emitted_test_Xy[1])

    assert np.allclose(emitted_df.values, transformer_df.values, atol=1e-5)


    



