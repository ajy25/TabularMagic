import pathlib
import sys
import pandas as pd
import pytest
import numpy as np


parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))


import tablemage as tm


@pytest.fixture
def setup_data():
    card_df = pd.read_csv(parent_dir / "demo" / "causal" / "data" / "card.csv")
    card_df["educ_binary"] = card_df["educ"].apply(lambda x: 0 if x < 13 else 1)
    return {
        "card": card_df,
    }


def test_naive(setup_data):
    data = setup_data["card"]
    analyzer = tm.Analyzer(data)
    causal_model = analyzer.causal(
        treatment="educ_binary", outcome="lwage", confounders=[]
    )
    report = causal_model.estimate_ate(method="naive")
    naive_estimate = report._estimate
    assert np.isclose(naive_estimate, 0.1942883, atol=1e-6)
    assert np.isclose(report._estimate_se, 0.01579299, atol=1e-6)


# def test_
