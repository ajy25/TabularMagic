import pandas as pd
import numpy as np
from ...preprocessing import CustomFunctionSingleVar


def parse_rlike_formula(formula: str):
    """Parses the formula portion of an R-like lm() call.
    
    Parameters
    ----------
    - formula : str. 
        An example is "y ~ log(x1) + poly(x2, 2)

    Returns
    -------
    - y_var : str
    - y_transform : str
    - X_vars : list[str]
    - X_transforms_dict : {str : list[str]}
    """
    formula = ''.join(formula.split(' '))
    y_var, X_formula = formula.split('~')
    y_transform = 'none'
    X_expressions = X_formula.split('+')
    X_vars = []
    X_transforms_dict = {
        'none': [],
    }

    for X_expression in X_expressions:

        # first check for log transform
        if X_expression[:3] == 'log':
            logvarname = X_expression[4:-1]
            if 'log' not in X_transforms_dict:
                X_transforms_dict['log'] = logvarname
            else:
                X_transforms_dict['log'].append(logvarname)
            X_vars.append(logvarname)
        
        # next check for poly transform
        elif X_expression[:4] == 'poly':
            within_paren = X_expression[5:-1]
            within_paren_split = within_paren.split(',')
            if len(within_paren_split) == 1:
                polyvarname = within_paren_split
            else:
                polyvarname, deg = within_paren_split
            deg = int(deg)
            if f'poly_{deg}' not in X_transforms_dict:
                X_transforms_dict[f'poly_{deg}'] = polyvarname
            else:
                X_transforms_dict[f'poly_{deg}'].append(polyvarname)
            X_vars.append(polyvarname)

        # all others are not to be transformed
        else:
            X_transforms_dict['none'].append(X_expression)
            X_vars.append(X_expression)

    # repeat for y_var
    if y_var[:3] == 'log':
        logvarname = y_var[4:-1]
        if 'log' not in X_transforms_dict:
            X_transforms_dict['log'] = logvarname
        else:
            X_transforms_dict['log'].append(logvarname)
        X_vars.append(logvarname)
    
    # next check for poly transform
    elif X_expression[:4] == 'poly':
        within_paren = X_expression[5:-1]
        within_paren_split = within_paren.split(',')
        if len(within_paren_split) == 1:
            polyvarname = within_paren_split
        else:
            polyvarname, deg = within_paren_split
        deg = int(deg)
        if f'poly_{deg}' not in X_transforms_dict:
            X_transforms_dict[f'poly_{deg}'] = polyvarname
        else:
            X_transforms_dict[f'poly_{deg}'].append(polyvarname)
        X_vars.append(polyvarname)

    # all others are not to be transformed
    else:
        X_transforms_dict['none'].append(X_expression)
        X_vars.append(X_expression)

    return y_var, y_transform, X_vars, X_transforms_dict





def transform_rlike(df: pd.DataFrame, y_var, y_transform, X_vars, 
                    X_transforms_dict):
    """Given outputs from parse_rlike_formula(), automatically 

    Parameters
    ----------
    - df : pd.DataFrame

    Returns
    -------
    - y_series : pd.Series
    - X_df : pd.DataFrame
    """
    pass


def parse_and_transform_rlike(formula: str, df: pd.DataFrame):
    """Transforms the data given a formula. Also one-hot-encodes categorical 
    variables and drops examples that contain missing entries. 

    Parameters
    ----------
    - df : pd.DataFrame

    Returns
    -------
    - y_series : pd.Series
    - X_df : pd.DataFrame
    """
    return transform_rlike(df, *parse_rlike_formula(formula))

        
        


