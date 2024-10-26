from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
from functools import partial
from .tooling_context import ToolingContext


# pandas query tool
class PandasQueryInput(BaseModel):
    query: str = Field(
        description="Query for extracting information from the dataset. "
        "The query must be in plain English (natural language)."
    )


def pandas_query_function(query: str, context: ToolingContext) -> str:
    response = context._data_container.pd_query_engine.query(query)
    return str(response)


def build_pandas_query_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(pandas_query_function, context=context),
        name="pandas_query_tool",
        description="Executes a natural language query on the user-provided DataFrame. "
        "Returns the response in natural language.",
        fn_schema=PandasQueryInput,
    )


# Get variable description tool
class _GetVariableDescriptionInput(BaseModel):
    var: str = Field(description="The variable to get the description of.")


def _get_variable_description_function(var: str, context: ToolingContext) -> str:
    return context._data_container.variable_info.get_description(var)


def build_get_variable_description_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_get_variable_description_function, context=context),
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


def _set_variable_description_function(
    var: str, description: str, context: ToolingContext
) -> str:
    context._data_container.variable_info.set_description(var, description)
    return f"Description of variable {var} has been set to: {description}."


def build_set_variable_description_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_set_variable_description_function, context=context),
        name="set_variable_description_tool",
        description="Sets the description of a variable. "
        "The 'get_variable_description_tool' tool can be used to retrieve the "
        "description at a later time.",
        fn_schema=_SetVariableDescriptionInput,
    )
