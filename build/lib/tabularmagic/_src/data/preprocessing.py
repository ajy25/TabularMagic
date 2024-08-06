import numpy as np
from typing import Callable


from sklearn.preprocessing import OneHotEncoder


class CustomOneHotEncoder(OneHotEncoder):
    def get_feature_names_out(self, categorical_vars):
        feature_names = super().get_feature_names_out(categorical_vars)
        custom_feature_names = []
        for feature_name in feature_names:
            split_name = feature_name.split("_")
            label = split_name[-1]
            var = "_".join(split_name[:-1])
            custom_feature_names.append(f"{label}_yn({var})")
        return custom_feature_names


class BaseSingleVarScaler:
    def __init__(self, var_name: str, x: np.ndarray):
        self.var_name = var_name
        self.x = x[~np.isnan(x)]
        self.fit()

    def fit(self):
        pass

    def transform(self, x: np.ndarray):
        """Transforms x. Robust to missing values in x."""
        pass

    def inverse_transform(self, x_scaled: np.ndarray):
        """Inverse transforms x_scaled. Robust to missing values in x_scaled."""
        pass


class MinMaxSingleVar(BaseSingleVarScaler):
    """Min max scaling of a single variable"""

    def __init__(self, var_name: str, x: np.ndarray):
        super().__init__(var_name, x)

    def fit(self):
        self.min = self.x.min()
        self.max = self.x.max()

    def transform(self, x: np.ndarray):
        """Transforms x. Robust to missing values in x."""
        return (x - self.min) / (self.max - self.min)

    def inverse_transform(self, x_scaled: np.ndarray):
        """Inverse transforms x_scaled. Robust to missing values in x_scaled."""
        return (self.max - self.min) * x_scaled + self.min


class StandardizeSingleVar(BaseSingleVarScaler):
    """Standard scaling of a single variable"""

    def __init__(self, var_name: str, x: np.ndarray):
        super().__init__(var_name, x)

    def fit(self):
        self.sigma = self.x.std()
        self.mu = self.x.mean()

    def transform(self, x: np.ndarray):
        """Transforms x. Robust to missing values in x."""
        return (x - self.mu) / self.sigma

    def inverse_transform(self, x_scaled: np.ndarray):
        """Inverse transforms x_scaled. Robust to missing values in x_scaled."""
        return self.sigma * x_scaled + self.mu


class LogTransformSingleVar(BaseSingleVarScaler):
    """Log (base e) transform scaling of a single variable"""

    def __init__(self, var_name: str, x: np.ndarray):
        super().__init__(var_name, x)

    def fit(self):
        pass

    def transform(self, x: np.ndarray):
        """Transforms x. Robust to missing values in x."""
        return np.log(x)

    def inverse_transform(self, x_scaled: np.ndarray):
        """Inverse transforms x_scaled. Robust to missing values in x_scaled."""
        return np.exp(x_scaled)


class ExpTransformSingleVar(BaseSingleVarScaler):
    """Exp (base e) transform scaling of a single variable"""

    def __init__(self, var_name: str, x: np.ndarray):
        super().__init__(var_name, x)

    def fit(self):
        pass

    def transform(self, x: np.ndarray):
        """Transforms x. Robust to missing values in x."""
        return np.exp(x)

    def inverse_transform(self, x_scaled: np.ndarray):
        """Inverse transforms x_scaled. Robust to missing values in x_scaled."""
        return np.log(x_scaled)


class Log1PTransformSingleVar(BaseSingleVarScaler):
    """Log1p transform scaling of a single variable"""

    def __init__(self, var_name: str, x: np.ndarray):
        super().__init__(var_name, x)

    def fit(self):
        pass

    def transform(self, x: np.ndarray):
        """Transforms x. Robust to missing values in x."""
        return np.log1p(x)

    def inverse_transform(self, x_scaled: np.ndarray):
        """Inverse transforms x_scaled. Robust to missing values in x_scaled."""
        return np.expm1(x_scaled)


class CustomFunctionSingleVar(BaseSingleVarScaler):
    """Custom scaling of a single variable"""

    def __init__(self, var_name: str, x: np.ndarray, f: Callable, f_inv: Callable):
        """
        Parameters
        ----------
        - var_name : str.
        - x : np.ndarray ~ (n_examples,).
        - f : function.
        f must have one argument, an 1d np.ndarray, and return an np.ndarray
        of the same size.
        - f_inv : function.
        Inverse function of f.
        """
        self.var_name = var_name
        self.x = x[~np.isnan(x)]
        self.f = f
        self.f_inv = f_inv
        self.fit()

    def fit(self):
        pass

    def transform(self, x: np.ndarray):
        return self.f(x)

    def inverse_transform(self, x_scaled: np.ndarray):
        return self.f_inv(x_scaled)
