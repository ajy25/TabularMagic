from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
from functools import partial
from json import dumps
from typing import Literal
from .tooling_context import ToolingContext
from .._debug.logger import print_debug
from ....ml import LinearC, LinearR, SVMC, SVMR, TreesC, TreesR, MLPC, MLPR
from ...._base import BaseC, BaseR


def parse_model_list_from_str(
    models_str: str, type: Literal["classification", "regression"]
) -> list[BaseC] | list[BaseR]:
    list_of_models = [model_str.strip() for model_str in models_str.split(",")]
    output = []
    if type == "regression":
        for model_str in list_of_models:
            if model_str == "OLS":
                output.append(LinearR("ols", name=model_str))
            elif model_str == "Ridge":
                output.append(LinearR("l2", name=model_str, n_jobs=2))
            elif model_str == "Lasso":
                output.append(LinearR("l1", name=model_str, n_jobs=2))
            elif model_str == "RF":
                output.append(TreesR("random_forest", name=model_str, n_jobs=2))
            elif model_str == "XGBoost":
                output.append(TreesR("xgboost", name=model_str, n_jobs=2))
            elif model_str == "SVM":
                output.append(SVMR("rbf", name=model_str, n_jobs=2))
            elif model_str == "MLP":
                output.append(MLPR(name=model_str, n_jobs=2))
            else:
                raise ValueError(f"Invalid model specification: {model_str}")
    elif type == "classification":
        for model_str in list_of_models:
            if model_str == "Logistic":
                output.append(LinearC("logistic", name=model_str))
            elif model_str == "Ridge":
                output.append(LinearC("l2", name=model_str))
            elif model_str == "Lasso":
                output.append(LinearC("l1", name=model_str))
            elif model_str == "RF":
                output.append(TreesC("random_forest", name=model_str))
            elif model_str == "XGBoost":
                output.append(TreesC("xgboost", name=model_str))
            elif model_str == "SVM":
                output.append(SVMC("rbf", name=model_str))
            elif model_str == "MLP":
                output.append(MLPC(name=model_str))
            else:
                raise ValueError(f"Invalid model specification: {model_str}")

    else:
        raise ValueError(f"Invalid ML problem type: {type}")

    return output


def parse_predictor_list_from_str(predictors_str: str) -> list[str]:
    return [predictor.strip() for predictor in predictors_str.split(",")]


class _MLRegressionInput(BaseModel):
    models: str = Field(
        description="""A comma delimited string of machine learning models to evaluate.
        The available models are (in `Model Name`: Description format)...

        1. `OLS`: Ordinary least squares regression
        2. `Ridge`: Linear regression with L2 penalty
        3. `Lasso`: Linear regression with L1 penalty
        4. `RF`: Random forest regressor
        5. `XGBoost`: XGBoost regressor
        6. `SVM`: Support vector machine regressor with radial basis function kernel
        7. `MLP`: Multilayer perceptron regressor

        An example input (without the quotes) is: `OLS, Lasso, XGBoost`.
        """
    )
    target: str = Field(
        description="The target variable, i.e. the variable to predict."
    )
    predictors: str = Field(
        description="""A comma delimited string of variables used by the models 
        to predict the target.

        An example input (without the quotes) is: `var1, var2, var3`.
        """
    )


def _ml_regression_function(
    models: str, target: str, predictors: str, context: ToolingContext
) -> str:
    models_list = parse_model_list_from_str(models, type="regression")
    print_debug(models_list)
    predictors_list = parse_predictor_list_from_str(predictors)
    print_debug(predictors_list)
    report = context._data_container.analyzer.regress(
        models=models_list, target=target, predictors=predictors_list
    )
    output_str = context._vectorstore_manager.add_str(dumps(report._to_dict()))
    return output_str


def build_ml_regression_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_ml_regression_function, context=context),
        name="ml_regression_tool",
        description="""Performs regression with a list of machine learning models that you must specify.
        Predicts the target variable with a list of predictor variables.
        Returns a string describing the model performances.
        The output string will be added to STORAGE.
        """,
        fn_schema=_MLRegressionInput,
    )


class _MLClassificationInput(BaseModel):
    models: str = Field(
        description="""A comma delimited string of machine learning models to evaluate.
        The available models are (in `Model Name`: Description format)...

        1. `Logistic`: Logistic regression
        2. `Ridge`: Logistic regression with L2 penalty
        3. `Lasso`: Logistic regression with L1 penalty
        4. `RF`: Random forest classifier
        5. `XGBoost`: XGBoost classifier
        6. `SVM`: Support vector machine classifier with radial basis function kernel
        7. `MLP`: Multilayer perceptron classifier

        An example input (without the quotes) is: `Logistic, RF, XGBoost`.
        """
    )
    target: str = Field(
        description="The target variable, i.e. the variable to predict."
    )
    predictors: str = Field(
        description="""A comma delimited string of variables used by the models 
        to predict the target.

        An example input (without the quotes) is: `var1, var2, var3`.
        """
    )


def _ml_classification_function(
    models: str, target: str, predictors: str, context: ToolingContext
) -> str:
    models_list = parse_model_list_from_str(models, type="classification")
    print_debug(models_list)
    predictors_list = parse_predictor_list_from_str(predictors)
    print_debug(predictors_list)
    report = context._data_container.analyzer.classify(
        models=models_list, target=target, predictors=predictors_list
    )
    output_str = context._vectorstore_manager.add_str(dumps(report._to_dict()))
    return output_str


def build_ml_classification_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_ml_classification_function, context=context),
        name="ml_classification_tool",
        description="""Performs classification with a list of machine learning models that you must specify. 
        Predicts the target variable with a list of predictor variables. 
        Returns a string describing the model performances.
        The output string will be added to STORAGE.
        """,
        fn_schema=_MLClassificationInput,
    )
