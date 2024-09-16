from langchain_core.tools import tool
from langchain.pydantic_v1 import BaseModel, Field

from ..tm import tabularwizard_analyzer


from ..io.jsontools import json_cache_tracker
from ..io.csvtools import csv_cache_tracker


class _TTestInput(BaseModel):
    categorical_var: str = Field(
        description="The binary categorical variable that defines the two groups."
    )
    numeric_var: str = Field(
        description="The numeric variable to compare between the two groups."
    )


@tool("ttest_tool", args_schema=_TTestInput, return_direct=True)
def ttest(categorical_var, numeric_var) -> str:
    """Conducts a t-test between two groups of data. Returns a JSON string
    containing the results of the t-test.
    """
    json_output = (
        tabularwizard_analyzer.eda("all")
        .ttest(numeric_var=numeric_var, stratify_by=categorical_var)
        ._to_json()
    )
    json_cache_tracker.save_to_json(json_output)
    return json_output
