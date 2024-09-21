from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field

from ..shared_tabularmagic import shared_container
from ..io.jsonutils import save_dict_to_json, save_df_to_json


# Means test tool
class TestEqualMeansTool(BaseModel):
    categorical_var: str = Field(
        description="The categorical variable that defines the groups/levels."
    )
    numeric_var: str = Field(
        description="The numeric variable to test between the groups/levels."
    )


def test_equal_means_function(categorical_var: str, numeric_var: str) -> str:
    """Tests whether the means of a numeric variable are equal across the different
    levels of a categorical variable."""
    dict_output = (
        shared_container.analyzer.eda("all")
        .test_equal_means(numeric_var=numeric_var, stratify_by=categorical_var)
        ._to_dict()
    )
    return save_dict_to_json(dict_output)


test_equal_means_tool = FunctionTool.from_defaults(
    fn=test_equal_means_function,
    name="test_equal_means_tool",
    description="Tests whether the means of a numeric variable are equal across "
    "the different levels of a categorical variable. "
    "The null hypothesis is that the means are equal. "
    "Automatically determines the correct statistical test to conduct. "
    "Returns a JSON string containing the results.",
    fn_schema=TestEqualMeansTool,
)


# Numerical summary stats tool
def numerical_summary_stats_function() -> str:
    """Calculates summary statistics for all numeric columns in the dataset."""
    df_output = shared_container.analyzer.eda("all").numeric_stats()
    return save_df_to_json(df_output)


numerical_summary_stats_tool = FunctionTool.from_defaults(
    fn=numerical_summary_stats_function,
    name="numerical_summary_stats_tool",
    description="Calculates summary statistics for all numeric columns in the dataset. "
    "Summary statistics include: count, 5 number summary, standard deviation, "
    "missing percentage, and the first, second, and third moments."
    "Returns a JSON string containing the summary statistics.",
)
