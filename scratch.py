from tabularmagic.linear.regression.lm_rlike_util import parse_and_transform_rlike
import numpy as np
import pandas as pd



df = pd.DataFrame(
        {
            'y': np.arange(10) + 1,
            'x1': np.arange(10) + 1,
            'x2': np.arange(10) + 1,
            'x3': np.arange(10) + 1,
            'x4': np.arange(10) + 1
        }
    )

y_series, y_scaler, X_df = parse_and_transform_rlike(
    'y ~ poly(x1, 2)', df)



print(y_series)
print(y_scaler)
print(X_df)






