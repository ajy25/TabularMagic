import pandas as pd

# Creating a DataFrame with an 'object' column
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'San Francisco', 'Los Angeles'],
    'Category': ['A', 'B', 'A']
}

df = pd.DataFrame(data)

# Display the DataFrame
print("DataFrame:")
print(df)

# Display data types of each column
print("\nData Types:")
print(df.dtypes)
