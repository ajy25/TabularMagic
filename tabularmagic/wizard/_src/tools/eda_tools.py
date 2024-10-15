from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field

from json import dumps

from ..datacontainer import GLOBAL_DATA_CONTAINER
from ..io.global_io import GLOBAL_IO


# Means test tool
class _TestEqualMeansInput(BaseModel):
    categorical_var: str = Field(
        description="The categorical variable that defines the groups/levels."
    )
    numeric_var: str = Field(
        description="The numeric variable to test between the groups/levels."
    )


def _test_equal_means_function(categorical_var: str, numeric_var: str) -> str:
    """Tests whether the means of a numeric variable are equal across the different
    levels of a categorical variable."""
    dict_output = (
        GLOBAL_DATA_CONTAINER.analyzer.eda("all")
        .test_equal_means(numeric_var=numeric_var, stratify_by=categorical_var)
        ._to_dict()
    )
    return GLOBAL_IO.add_str(dumps(dict_output))


test_equal_means_tool = FunctionTool.from_defaults(
    fn=_test_equal_means_function,
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


def _plot_distribution_function(var: str) -> str:
    """Plots the distribution of a variable."""
    fig = GLOBAL_DATA_CONTAINER.analyzer.eda("all").plot_distribution(var)
    return GLOBAL_IO.add_figure(fig, f"Distribution plot of variable: {var}. ")


plot_distribution_tool = FunctionTool.from_defaults(
    fn=_plot_distribution_function,
    name="plot_distribution_tool",
    description="Plots the distribution of a variable. "
    "A detailed text description of the plot will be saved to STORAGE, "
    "along with the plot itself. "
    "The text description will also be returned.",
    fn_schema=_PlotDistributionInput,
)


# Numerical summary statistics tool
def _numerical_summary_statistics_function() -> str:
    """Generates numerical summary statistics for the dataset."""
    dict_output = GLOBAL_DATA_CONTAINER.analyzer.eda("all").numeric_stats().to_dict()
    return GLOBAL_IO.add_str(dumps(dict_output))


numeric_summary_statistics_tool = FunctionTool.from_defaults(
    fn=_numerical_summary_statistics_function,
    name="numerical_summary_statistics_tool",
    description="Generates summary statistics for the numeric variables in the dataset. "
    "Returns a JSON string containing the summary statistics. "
    "The JSON string will be added to STORAGE.",
)


# Categorical summary statistics tool
def _categorical_summary_statistics_function() -> str:
    """Generates categorical summary statistics for the dataset."""
    dict_output = (
        GLOBAL_DATA_CONTAINER.analyzer.eda("all").categorical_stats().to_dict()
    )
    return GLOBAL_IO.add_str(dumps(dict_output))


categorical_summary_statistics_tool = FunctionTool.from_defaults(
    fn=_categorical_summary_statistics_function,
    name="categorical_summary_statistics_tool",
    description="Generates summary statistics for the categorical variables in the dataset. "
    "Returns a JSON string containing the summary statistics. "
    "The JSON string will be added to STORAGE.",
)
