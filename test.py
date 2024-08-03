import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

# Example transformation function
def custom_transformer(df):
    # Assume df is a DataFrame
    df['new_col'] = df['existing_col'] * 2
    return df

# Create a FunctionTransformer
transformer = FunctionTransformer(custom_transformer, validate=False)

# Example DataFrame
df = pd.DataFrame({'existing_col': [1, 2, 3]})

# Apply the transformer
transformed_df = transformer.transform(df)
print(transformed_df)

# Create a pipeline with the transformer
pipeline = Pipeline([
    ('custom', transformer)
])

# Fit and transform the pipeline (fit might be a no-op depending on the transformer)
transformed_df = pipeline.fit_transform(df)
print(transformed_df)