import pandas as pd

def one_hot_encode_categorical(df, categorical_variable):
    """
    One-hot encodes a categorical variable in a DataFrame and drops the original column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - categorical_variable (str): The name of the categorical variable to be one-hot encoded.

    Returns:
    - pd.DataFrame: The DataFrame with the specified variable one-hot encoded and the original column dropped.
    - list: The list of new variable names after one-hot encoding.
    """
    # Ensure the specified variable is categorical
    if df[categorical_variable].dtype.name != 'category':
        df[categorical_variable] = df[categorical_variable].astype('category')

    # Perform one-hot encoding
    df_encoded = pd.get_dummies(df, columns=[categorical_variable], drop_first=True)

    # Get the list of new variable names
    new_variable_names = list(df_encoded.columns)

    return df_encoded, new_variable_names

# Example usage:
data = {'feature1': [1, 2, 3, 4, 5],
        'category': ['A', 'B', 'A', 'C', 'B']}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

# Apply one-hot encoding and get the new variable names
df_encoded, new_variable_names = one_hot_encode_categorical(df, 'category')

print("\nDataFrame after one-hot encoding:")
print(df_encoded)

print("\nNew variable names after one-hot encoding:")
print(new_variable_names)
