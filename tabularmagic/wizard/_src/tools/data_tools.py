from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field

from ..datacontainer import GLOBAL_DATA_CONTAINER


# pandas query tool
class PandasQueryInput(BaseModel):
    query: str = Field(
        description="Query for extracting information from the dataset. "
        "The query must be in plain English (natural language)."
    )


def pandas_query_function(query: str) -> str:
    """Executes a natural language query on the user-provided DataFrame.
    The output is also in natural language.
    """
    response = GLOBAL_DATA_CONTAINER.pd_query_engine.query(query)
    return str(response)


pandas_query_tool = FunctionTool.from_defaults(
    fn=pandas_query_function,
    name="pandas_query_tool",
    description="Executes a natural language query on the user-provided DataFrame. "
    "Returns the response in natural language.",
    fn_schema=PandasQueryInput,
)


# Get variable description tool
class _GetVariableDescriptionInput(BaseModel):
    var: str = Field(description="The variable to get the description of.")


def _get_variable_description_function(var: str) -> str:
    """Gets the description of a variable."""
    return GLOBAL_DATA_CONTAINER.variable_info.get_description(var)


get_variable_description_tool = FunctionTool.from_defaults(
    fn=_get_variable_description_function,
    name="get_variable_description_tool",
    description="Gets the description of a variable. "
    "The description will be returned. "
    "If no description is available, an empty string will be returned.",
    fn_schema=_GetVariableDescriptionInput,
)


# Set variable description tool
class _SetVariableDescriptionInput(BaseModel):
    var: str = Field(description="The variable to set the description of.")
    description: str = Field(description="The description of the variable.")


def _set_variable_description_function(var: str, description: str) -> str:
    """Sets the description of a variable."""
    GLOBAL_DATA_CONTAINER.variable_info.set_description(var, description)
    return f"Description of variable {var} set to: {description}. "


set_variable_description_tool = FunctionTool.from_defaults(
    fn=_set_variable_description_function,
    name="set_variable_description_tool",
    description="Sets the description of a variable. "
    "The 'get_variable_description_tool' tool can be used to retrieve the description "
    "at a later time.",
    fn_schema=_SetVariableDescriptionInput,
)
