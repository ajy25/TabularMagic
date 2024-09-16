from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tm import tabularwizard_analyzer


from ..io.jsonutils import save_dict_to_json, save_df_to_json


# T-test tool.
class _TTestInput(BaseModel):
    categorical_var: str = Field(
        description="The binary categorical variable that defines the two groups."
    )
    numeric_var: str = Field(
        description="The numeric variable to compare between the two groups."
    )


@tool("ttest_tool", args_schema=_TTestInput)
def ttest_tool(categorical_var: str, numeric_var: str) -> str:
    """Conducts a t-test between two groups of data. Returns a JSON string
    containing the results of the t-test.
    """
    dict_output = (
        tabularwizard_analyzer.eda("all")
        .ttest(numeric_var=numeric_var, stratify_by=categorical_var)
        ._to_dict()
    )
    return save_dict_to_json(dict_output)


# Numerical summary stats tool.
@tool("numerical_summary_stats_tool")
def numerical_summary_stats_tool() -> str:
    """Calculates summary statistics for all numeric columns in the dataset.
    Returns a JSON string containing the summary statistics.
    """
    df_output = tabularwizard_analyzer.eda("all").numeric_stats()
    return save_df_to_json(df_output)
