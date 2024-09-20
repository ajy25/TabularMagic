import sys
import pathlib
import numpy as np
import pandas as pd
import pytest

parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))


import tabularmagic as tm




@pytest.fixture
def setup_data():
    np.random.seed(42)
    X = pd.DataFrame({
        'x1': np.random.randn(100) * 0.5 + 10,    # Continuous variable
        'x2': np.random.randn(100) * 3,    # Continuous variable (noise)
        'cat1': np.random.choice(['A', 'B', 'C'], 100),  # Categorical variable
        'cat2': np.random.choice(['W', 'X', 'Y', 'Z'], 100),  # Categorical variable (noise)
        'x3': np.random.randn(100) * 3     # Continuous variable (noise) 
    })
    
    # Dependent variable
    y = 3 * X['x1'] - 5 * (X['cat1'] == 'A') + 10 * (X['cat1'] == 'B') + np.random.randn(100)

    df_ols = X
    df_ols['y'] = y

    return {
        'df_ols': df_ols
    }


def test_backward_selection_ols(setup_data):
    df = setup_data['df_ols']

    analyzer = tm.Analyzer(
        df=df
    )
    lmreport = analyzer.lm(
        target='y',
        predictors=['x1', 'x2', 'cat1', 'cat2', 'x3'],
    )
    reduced_lmreport = lmreport.step('backward')

    selected_features = reduced_lmreport.train_report()._X_eval_df.columns.tolist()

    print(selected_features)
    
    # Assert that only the most important features are selected
    assert 'x1' in selected_features  # Should be included
    assert 'x2' not in selected_features  # Should be excluded
    assert 'x3' not in selected_features  # Should be excluded
    
    # All cat2 levels should be excluded
    # All cat1 levels should be included, with the exception of 'A' (we drop first)
    for feature in selected_features:
        if 'cat2' in feature:
            assert False
        if 'cat1' in feature:
            assert feature != 'cat1::A'

        

def test_forward_selection_ols(setup_data):
    df = setup_data['df_ols']

    analyzer = tm.Analyzer(
        df=df
    )
    lmreport = analyzer.lm(
        target='y',
        predictors=['x1', 'x2', 'cat1', 'cat2', 'x3'],
    )
    reduced_lmreport = lmreport.step('forward')

    selected_features = reduced_lmreport.train_report()._X_eval_df.columns.tolist()

    print(selected_features)
    
    # Assert that only the most important features are selected
    assert 'x1' in selected_features  # Should be included
    assert 'x2' not in selected_features  # Should be excluded
    assert 'x3' not in selected_features  # Should be excluded
    
    # All cat2 levels should be excluded
    # All cat1 levels should be included, with the exception of 'A' (we drop first)
    for feature in selected_features:
        if 'cat2' in feature:
            assert False
        if 'cat1' in feature:
            assert feature != 'cat1::A'



def test_both_direction_selection_ols(setup_data):
    df = setup_data['df_ols']

    analyzer = tm.Analyzer(
        df=df
    )
    lmreport = analyzer.lm(
        target='y',
        predictors=['x1', 'x2', 'cat1', 'cat2', 'x3'],
    )
    reduced_lmreport = lmreport.step('both')

    selected_features = reduced_lmreport.train_report()._X_eval_df.columns.tolist()

    print(selected_features)
    
    # Assert that only the most important features are selected
    assert 'x1' in selected_features  # Should be included
    assert 'x2' not in selected_features  # Should be excluded
    assert 'x3' not in selected_features  # Should be excluded
    
    # All cat2 levels should be excluded
    # All cat1 levels should be included, with the exception of 'A' (we drop first)
    for feature in selected_features:
        if 'cat2' in feature:
            assert False
        if 'cat1' in feature:
            assert feature != 'cat1::A'

