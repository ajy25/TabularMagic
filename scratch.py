import pandas as pd

# Creating a DataFrame with an 'object' column
data = {
    'New York': [1, 0, 0],
    'San Francisco': [0, 1, 0],
    'Los Angeles': [0, 0, 1]
}

df = pd.DataFrame(data)

# Display the DataFrame
print("DataFrame:")
print(df)

# Display data types of each column
print(df.idxmax(axis=1))



