import pandas as pd
import numpy as np

# Create a numerical DataFrame with random data
np.random.seed(0)  # For reproducibility
data = {
    'A': np.random.rand(10),  # 10 random numbers for column A
    'B': np.random.rand(10),  # 10 random numbers for column B
    'C': np.random.rand(10),  # 10 random numbers for column C
    'D': np.random.rand(10)   # 10 random numbers for column D
}

df = pd.DataFrame(data)

vars = df.select_dtypes(include=['object', 'category', 'bool']).columns.to_list()

print(vars)
