import pandas as pd
from typing import Literal
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from ...data import DataEmitter


def score_model(
    emitter: DataEmitter,
    feature_list: list[str],
    model: Literal["ols", "logit"],
    metric: Literal["aic", "bic"],
    y_label_encoder: LabelEncoder | None = None,
) -> float:
    """Scores a linear model.

    Parameters
    ----------
    emitter : DataEmitter
        The data emitter.

    feature_list : list[str]
        The list of features to use in the model. These should be
        PRE-one-hot encoded features.

    model : Literal["ols", "logit"]
        The model to use.

    metric : Literal["aic", "bic"]
        The metric to use for scoring.

    y_label_encoder : LabelEncoder | None
        The label encoder for the target variable, by default None.
        Only used for the binomial model.
    """
    if len(feature_list) == 0:
        return np.inf

    # obtain the data
    emitter.select_predictors_pre_onehot(feature_list)
    X_train, y_train = emitter.emit_train_Xy()

    # like typical fitting, we enforce a constant, regardless of prior existence
    X_train_w_constant = sm.add_constant(X_train, has_constant="add")

    # fit the appropriate model, no need for heterscedasticity robust standard errors
    if model == "ols":
        new_model = sm.OLS(y_train, X_train_w_constant)
    elif model == "logit":
        if y_label_encoder is not None:
            y_train = y_label_encoder.transform(y_train)
        new_model = sm.Logit(y_train, X_train_w_constant)
    else:
        raise ValueError("Model must be one of 'ols', 'logit'.")

    if metric == "aic":
        return new_model.fit().aic
    elif metric == "bic":
        return new_model.fit().bic
    else:
        raise ValueError("Metric must be one of 'aic', 'bic'")
