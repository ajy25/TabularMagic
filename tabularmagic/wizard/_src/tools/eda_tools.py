from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
from json import dumps
from functools import partial
from .tooling_context import ToolingContext


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
    return context._vectorstore_manager.add_str(dumps(dict_output))


def build_test_equal_means_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_test_equal_means_function, context=context),
        name="test_equal_means_tool",
        description="Tests whether the means of a numeric variable are equal across "
        "the different levels of a categorical variable. "
        "The null hypothesis is that the means are equal. "
        "This tool will automatically determine the correct statistical test to conduct. "
        "Returns a JSON string containing results and which test used. "
        "The JSON string will be added to STORAGE.",
        fn_schema=_TestEqualMeansInput,
    )


# Plot distribution tool
class _PlotDistributionInput(BaseModel):
    var: str = Field(description="The variable to plot the distribution of.")


def _plot_distribution_function(var: str, context: ToolingContext) -> str:
    fig = context._data_container.analyzer.eda("all").plot_distribution(var)
    return context._vectorstore_manager.add_figure(
        fig, f"Distribution plot of variable: {var}."
    )


def build_plot_distribution_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_plot_distribution_function, context=context),
        name="plot_distribution_tool",
        description="Plots the distribution of a variable. "
        "A detailed text description of the plot will be saved to STORAGE, "
        "along with the plot itself. "
        "The text description will also be returned.",
        fn_schema=_PlotDistributionInput,
    )


# Numeric summary statistics tool
def _numeric_summary_statistics_function(context: ToolingContext) -> str:
    dict_output = context._data_container.analyzer.eda("all").numeric_stats().to_dict()
    return context._vectorstore_manager.add_str(dumps(dict_output))


def build_numeric_summary_statistics_tool(context: ToolingContext) -> FunctionTool:
    def temp_fn() -> str:
        return _numeric_summary_statistics_function(context)

    return FunctionTool.from_defaults(
        fn=temp_fn,
        name="numeric_summary_statistics_tool",
        description="Generates summary statistics for the numeric variables in the dataset. "
        "Returns a JSON string containing the summary statistics. "
        "The JSON string will be added to STORAGE.",
    )


# Categorical summary statistics tool
def _categorical_summary_statistics_function(context: ToolingContext) -> str:
    """Generates categorical summary statistics for the dataset."""
    dict_output = (
        context._data_container.analyzer.eda("all").categorical_stats().to_dict()
    )
    return context._vectorstore_manager.add_str(dumps(dict_output))


def build_categorical_summary_statistics_tool(context: ToolingContext) -> FunctionTool:
    def temp_fn() -> str:
        return _categorical_summary_statistics_function(context)

    return FunctionTool.from_defaults(
        fn=temp_fn,
        name="categorical_summary_statistics_tool",
        description="Generates summary statistics for the categorical variables in the dataset. "
        "Returns a JSON string containing the summary statistics. "
        "The JSON string will be added to STORAGE.",
    )
