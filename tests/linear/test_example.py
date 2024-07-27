import pathlib
import sys
import pandas as pd
import pytest
import numpy as np
import statsmodels.api as sm


parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

import tabularmagic as tm




df_mtcars = pd.read_csv(
    parent_dir / "demo" / "regression" / "mtcars_data" / "mtcars.csv"
)

  



def math_linear_regression(X, y):
    # Add a constant term (intercept) to the predictors
    X = np.column_stack((np.ones(X.shape[0]), X))
    
    # Compute the coefficients using the normal equation: (X.T * X)^-1 * X.T * y
    coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
    return coefficients

#def statsmodels_regression(data)

def tm_linear_regression(data):
    analyzer = tm.Analyzer(data, test_size=0.0, verbose=False)
    report = analyzer.lm(
        formula="mpg ~ cyl + disp"
    )
    extract_model = report._model.estimator
    return extract_model.params.iloc[0]

# Test case to compare the outputs
def test_function():
    # Get the output from the custom Python function
    tm_output = tm_linear_regression(df_mtcars)

    # Define the predictors and the response variable
    X = df_mtcars[['cyl', 'disp']]
    y = df_mtcars['mpg']
    
    # Add a constant term for the intercept
    X = sm.add_constant(X)
    
    # Fit the model using statsmodels
    model = sm.OLS(y, X).fit()
    
    # Get the first coefficient (the intercept)
    statsmodels_output = model.params.iloc[0]
    
    # Use assert to compare the outputs
    assert np.isclose(tm_output, statsmodels_output), f"Outputs do not match: {tm_output} != {statsmodels_output}"

# Run the test
if __name__ == "__main__":
    pytest.main([__file__])
