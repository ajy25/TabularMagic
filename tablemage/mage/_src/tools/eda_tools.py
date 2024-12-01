from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
from functools import partial
from .tooling_context import ToolingContext

from .._debug.logger import print_debug


# Means test tool
class _TestEqualMeansInput(BaseModel):
    categorical_var: str = Field(
        description="The categorical variable that defines the groups/levels."
    )
    numeric_var: str = Field(
        description="The numeric variable to test between the groups/levels."
    )


def _test_equal_means_function(
    categorical_var: str, numeric_var: str, context: ToolingContext
) -> str:
    dict_output = (
        context._data_container.analyzer.eda("all")
        .test_equal_means(numeric_var=numeric_var, stratify_by=categorical_var)
        ._to_dict()
    )
    context.add_thought(
        "I am going to test whether the means of the variable {numeric_var} are equal across the different levels of the variable {categorical_var}.".format(
            numeric_var=numeric_var, categorical_var=categorical_var
        )
    )
    context.add_code(
        "analyzer.eda().test_equal_means(numeric_var='{}', stratify_by='{}')".format(
            numeric_var, categorical_var
        )
    )
    return context.add_dict(dict_output)


def build_test_equal_means_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_test_equal_means_function, context=context),
        name="test_equal_means_tool",
        description="""Tests whether the means of a numeric variable are equal across the different levels of a categorical variable.
        The null hypothesis is that the means are equal.
        This tool will automatically determine the correct statistical test to conduct.
        Returns a JSON string containing results and which test used.""",
        fn_schema=_TestEqualMeansInput,
    )


# Plot distribution tool
class _PlotDistributionInput(BaseModel):
    var: str = Field(description="The variable to plot the distribution of.")


def _plot_distribution_function(var: str, context: ToolingContext) -> str:
    fig = context._data_container.analyzer.eda("all").plot_distribution(var)
    context.add_thought(
        "I am going to plot the distribution of the variable: {var}.".format(var=var)
    )
    context.add_code("analyzer.eda().plot_distribution('{}')".format(var))
    return context.add_figure(
        fig, text_description=f"Distribution plot of variable: {var}."
    )


def build_plot_distribution_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_plot_distribution_function, context=context),
        name="plot_distribution_tool",
        description="""Plots the distribution of variable.
        The text description will be returned.""",
        fn_schema=_PlotDistributionInput,
    )


# Correlation comparison tool
class _CorrelationComparisonInput(BaseModel):
    target: str = Field(description="The target variable to compare correlations with.")
    numeric_vars: str = Field(
        description="Comma delimited list of variables with which to compare correlations with the target variable.",
    )


def _correlation_comparison_function(
    target: str, numeric_vars: str, context: ToolingContext
) -> str:
    vars_list = [var.strip() for var in numeric_vars.split(",")]
    df_output = context.data_container.analyzer.eda(
        "all"
    ).tabulate_correlation_comparison(
        target=target,
        numeric_vars=vars_list,
    )
    context.add_thought(
        "I am going to compare the correlation of {target} with the following variables: {vars_list}.".format(
            target=target, vars_list=", ".join(vars_list)
        )
    )
    context.add_code(
        "analyzer.eda().tabulate_correlation_comparison(target='{}', numeric_vars=['{}'])".format(
            target, "', '".join(vars_list)
        )
    )
    return context.add_table(df_output, add_to_vectorstore=True)


def build_correlation_comparison_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_correlation_comparison_function, context=context),
        name="correlation_comparison_tool",
        description="Compares the correlation of a target variable with other numeric variables. "
        "Returns a JSON string containing the correlation values.",
        fn_schema=_CorrelationComparisonInput,
    )


# Correlation matrix tool
class _CorrelationMatrixInput(BaseModel):
    numeric_vars: str = Field(
        description="Comma delimited list of numeric variables to include in the correlation matrix. "
    )


def _correlation_matrix_function(numeric_vars: str, context: ToolingContext) -> str:
    df_output = context._data_container.analyzer.eda("all").tabulate_correlation_matrix(
        numeric_vars=[var.strip() for var in numeric_vars.split(",")]
    )
    context.add_thought(
        "I am going to compute the correlation matrix for the following variables: {vars_list}.".format(
            vars_list=numeric_vars
        )
    )
    context.add_code(
        "analyzer.eda().tabulate_correlation_matrix(numeric_vars=['{}'])".format(
            "', '".join([numeric_vars])
        )
    )
    return context.add_table(df_output, add_to_vectorstore=True)


def build_correlation_matrix_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_correlation_matrix_function, context=context),
        name="correlation_matrix_tool",
        description="""Computes a correlation matrix for the specified numeric variables.
        Returns a JSON string containing the correlation matrix.""",
        fn_schema=_CorrelationMatrixInput,
    )


class _BlankInput(BaseModel):
    pass


# Numeric summary statistics tool
def _numeric_summary_statistics_function(context: ToolingContext) -> str:
    df_output = context._data_container.analyzer.eda("all").numeric_stats()
    context.add_thought(
        "I am going to generate summary statistics for the numeric variables in the dataset."
    )
    context.add_code("analyzer.eda().numeric_stats()")
    return context.add_table(df_output, add_to_vectorstore=True)


def build_numeric_summary_statistics_tool(context: ToolingContext) -> FunctionTool:
    def temp_fn():
        return _numeric_summary_statistics_function(context)

    return FunctionTool.from_defaults(
        fn=temp_fn,
        name="numeric_summary_statistics_tool",
        description="""Generates summary statistics for the numeric variables in the dataset.
        Returns a JSON string containing the summary statistics.""",
        fn_schema=_BlankInput(),
    )


# Categorical summary statistics tool
def _categorical_summary_statistics_function(context: ToolingContext) -> str:
    """Generates categorical summary statistics for the dataset."""
    df_output = context._data_container.analyzer.eda("all").categorical_stats()
    context.add_thought(
        "I am going to generate summary statistics for the categorical variables in the dataset."
    )
    context.add_code("analyzer.eda().categorical_stats()")
    return context.add_table(df_output, add_to_vectorstore=True)


def build_categorical_summary_statistics_tool(context: ToolingContext) -> FunctionTool:
    def temp_fn():
        return _categorical_summary_statistics_function(context)

    return FunctionTool.from_defaults(
        fn=temp_fn,
        name="categorical_summary_statistics_tool",
        description="""Generates summary statistics for the categorical variables in the dataset.
        Returns a JSON string containing the summary statistics.""",
        fn_schema=_BlankInput(),
    )
