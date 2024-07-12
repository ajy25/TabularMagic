import pandas as pd
import numpy as np
from scipy import stats
import itertools
from ...data.preprocessing\
    import LogTransformSingleVar, CustomFunctionSingleVar


def is_continuous(var: str, df: pd.DataFrame):
    """
    Checks if a variable in a DataFrame is continuous.
    
    Parameters
    ----------
    - var : str. The name of the column to be checked.
    - df : pd.DataFrame. The DataFrame containing the variable.
    
    Returns
    -------
    - bool. True if the variable is continuous, False otherwise.
    """
    dtype = df[var].dtype
    if pd.api.types.is_numeric_dtype(dtype):
        if not pd.api.types.is_integer_dtype(dtype):
            return True
    return False

def poly(x: np.ndarray, degree: int):
    """Transforms x into orthogonal polynomials.
    
    Parameters
    ----------
    - x : np.ndarray ~ (n)
    - degree : int

    Returns
    -------
    - x_transformed : np.ndarray ~ (n, degree)
    """
    assert len(x.shape) == 1
    x = np.array(x)
    x_not_nan_idx = np.logical_not(np.isnan(x))
    output = np.full((len(x), degree), np.nan)
    X = np.transpose(np.vstack([x[x_not_nan_idx]**k for k in range(degree+1)]))
    output[x_not_nan_idx] = np.linalg.qr(X)[0][:,1:]
    return output

def check_all_parentheses(list_of_text):
    check = True
    for text in list_of_text:
        if not check:
            return check
        check = check and check_parentheses(text)
    return check

def check_parentheses(text):
    stack = []
    for char in text:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if not stack:
                return False
            stack.pop()
    return len(stack) == 0



def recursive_expression_transformer(expression: str, df: pd.DataFrame):
    """Handles a predictor expression. Handles interaction terms. Handles
    log transforms. 

    Parameters
    ----------
    - expression : str. 
    - df : pd.DataFrame. 

    Returns
    -------
    - pd.DataFrame.
    """

    # efficient multiplication checker for most cases
    if ':' in expression and check_all_parentheses(expression.split(':')):
        expression_split = expression.split(':')
        dfs_to_multiply = []
        for subexpression in expression_split:
            temp_df: pd.DataFrame =\
                recursive_expression_transformer(subexpression, df)
            dfs_to_multiply.append(
                temp_df
            )
        output = dfs_to_multiply[0]
        for i in range(1, len(dfs_to_multiply)):
            first_df = output
            last_df = dfs_to_multiply[i]
            cartesian_product = list(itertools.product(
                first_df.columns.to_list(), last_df.columns.to_list()))
            output_temp = dict()
            for var_a, var_b in cartesian_product:
                output_temp[f'{var_a}:{var_b}'] = first_df[var_a].to_numpy() * \
                    last_df[var_b].to_numpy()
            output = pd.DataFrame(output_temp, index=df.index)
        return output

    elif expression[:4] == 'log(' and check_parentheses(expression[4:-1]):
        output_df: pd.DataFrame =\
            recursive_expression_transformer(expression[4:-1], df)
        new_col_names = []
        for col in output_df.columns:
            new_col_names.append(f'log({col})')
            output_df.loc[:, col] = np.log(output_df[col].to_numpy())
        output_df.columns = new_col_names
        return output_df
    
    elif expression[:4] == 'exp(' and check_parentheses(expression[4:-1]):
        output_df: pd.DataFrame =\
            recursive_expression_transformer(expression[4:-1], df)
        new_col_names = []
        for col in output_df.columns:
            new_col_names.append(f'exp({col})')
            output_df.loc[:, col] = np.exp(output_df[col].to_numpy())
        output_df.columns = new_col_names
        return output_df
    
    elif expression[:5] == 'poly(' and check_parentheses(expression[5:-1]):
        within_paren = expression[5:-1]
        within_paren_split = within_paren.split(',')
        subexpression = ','.join(within_paren_split[:-1])
        if len(within_paren_split) < 2:
            raise ValueError('Error in formula when parsing poly(). ' + \
                             'Ensure degree is given as input.')
        deg_info = within_paren_split[-1]
        deg_info_split = deg_info.split('=')
        if len(deg_info_split) == 2 and deg_info_split[0] == 'degree':
            degree = int(deg_info_split[1])
        elif len(deg_info_split) == 1 and len(deg_info_split[0]) > 0:
            degree = int(deg_info_split[0])
        else:
            raise ValueError('Error in formula when parsing poly().')
        temp_df: pd.DataFrame =\
            recursive_expression_transformer(subexpression, df)
        output_dfs = []
        for col in temp_df.columns:
            output_dfs.append(
                pd.DataFrame(poly(temp_df[col].to_numpy(), degree), 
                    columns=[f'poly({col},{degree}){i+1}' \
                            for i in range(degree)], index=df.index)
            )
        return pd.concat(output_dfs, axis=1)
    
    # a less efficient multiplication-handling scheme for edge cases, only  
    # reaches this point if a transformation is being applied to first term
    elif ':' in expression:
        expression_split = expression.split(':')
        for right_cutoff in range(1, len(expression_split)):
            left_subexpression = ':'.join(expression_split[:right_cutoff])
            right_subexpression = ':'.join(expression_split[right_cutoff:])
            if not check_all_parentheses(
                [left_subexpression, right_subexpression]):
                continue
            first_df: pd.DataFrame =\
                recursive_expression_transformer(left_subexpression, df)
            last_df: pd.DataFrame =\
                recursive_expression_transformer(right_subexpression, df)
            cartesian_product = list(
                itertools.product(first_df.columns.to_list(), 
                                  last_df.columns.to_list()))
            output = dict()
            for var_a, var_b in cartesian_product:
                output[f'{var_a}:{var_b}'] = first_df[var_a].to_numpy() * \
                    last_df[var_b].to_numpy()
            return pd.DataFrame(output, index=df.index)
        raise ValueError('Error in formula when parsing poly().')

    else:
        if is_continuous(expression, df):
            return df[[expression]]
        else:
            output = pd.get_dummies(df[[expression]], drop_first=True)
            return output.astype(int)
        


def parse_and_transform_rlike(formula: str, df: pd.DataFrame):
    """Transforms the data given a formula. Also one-hot-encodes categorical 
    predictor variables and drops examples that contain missing entries. 

    Works for the following transformations:
    1. log()
    2. poly()
    3. boxcox()
    4. interactions (:)

    Parameters
    ----------
    - formula : str.
    - df : pd.DataFrame.

    Returns
    -------
    - y_series : pd.Series
    - y_scaler : None | LogTransformSingleVar
    - X_df : pd.DataFrame
    """
    # remove all spaces
    formula = ''.join(formula.split(' '))

    # identify the predictors and response portions of the formula
    y_formula, X_formula = formula.split('~')

    # y_var can only be log transformed
    if y_formula[:4] == 'log(':
        y_var = y_formula[4:-1]
        y_scaler = LogTransformSingleVar(var_name=y_var, x=df[y_var].to_numpy())
        y_series = pd.Series(y_scaler.transform(df[y_var].to_numpy()), 
                             name=y_var, index=df.index)
    elif y_formula[:7] == 'boxcox(':
        y_var = y_formula[7:-1]
        y_vals = df[y_var].to_numpy()
        _, lmda = stats.boxcox(y_vals)
        if lmda != 0:
            y_scaler = CustomFunctionSingleVar(var_name=y_var, x=y_vals, 
                f=lambda x: (x ** lmda - 1) / lmda, 
                f_inv=lambda x: (x * lmda + 1) ** (1 / lmda))
        else:
            y_scaler = LogTransformSingleVar(var_name=y_var, x=y_vals)
        y_series = pd.Series(y_scaler.transform(df[y_var].to_numpy()), 
                             name=y_var, index=df.index)
    else:
        y_var = y_formula
        y_scaler = None
        y_series = pd.Series(df[y_var].to_numpy(), name=y_var, index=df.index)

    # for the predictors protion of the formula, split on plus signs
    X_expressions = X_formula.split('+')
    X_dfs = []
    for X_expression in X_expressions:
        # recursively parse each expression
        X_dfs.append(recursive_expression_transformer(X_expression, df))

    # convert the list of series to a dataframe
    X_df = pd.concat(X_dfs, axis=1)
    return y_series, y_scaler, X_df
    

        





