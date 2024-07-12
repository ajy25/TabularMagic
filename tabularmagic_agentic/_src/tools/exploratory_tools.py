from langchain.tools import tool
from langchain.pydantic_v1 import BaseModel, Field
from ..shared_analyzer import shared_analyzer


class _TestEqualMeans_ToolArgs(BaseModel):
    """Arguments for the ttest_tool."""

    numerical_var: str = Field(description="The numerical variable of interest.")
    stratify_by: str = Field(description="The grouping variable.")


@tool("test_equal_means_tool", args_schema=_TestEqualMeans_ToolArgs)
def test_equal_means_tool(numerical_var: str, stratify_by: str) -> str:
    """Tests for equal means (null hypothesis) between two or more groups.
    Returns statistical test results (hypotheses, p-value, etc.) as a
    json-formatted string.
    """
    return (
        shared_analyzer.get_shared_analyzer()
        .eda()
        .test_equal_means(numerical_var=numerical_var, stratify_by=stratify_by)
        ._agentic_describe_json_str()
    )
