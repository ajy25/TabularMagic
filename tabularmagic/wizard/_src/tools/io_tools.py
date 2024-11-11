from llama_index.core.tools import FunctionTool
from llama_index.core.schema import BaseNode
from pydantic import BaseModel, Field
from functools import partial
from .tooling_context import ToolingContext


# save text tool
class _WriteTextInput(BaseModel):
    text: str = Field(description="The text to write to STORAGE.")


def _write_text_function(text: str, context: ToolingContext) -> str:
    context._vectorstore_manager.add_str(text)
    return "Text has been written to STORAGE."


def build_write_text_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_write_text_function, context=context),
        name="write_text_tool",
        description="Writes text to STORAGE. "
        "This tool is useful for storing summaries of results for later reference.",
        fn_schema=_WriteTextInput,
    )


# retrieve text tool
class _RetrieveTextOutput(BaseModel):
    query: str = Field(
        description="Natural language query to retrieve information in STORAGE. "
        "For example, if STORAGE contains summary statistics, an appropriate query "
        "could be: What is the mean of the variable 'mpg'?."
    )


def _retrieve_text_function(query: str, context: ToolingContext) -> str:
    retrieved_node: BaseNode = context._vectorstore_manager.retriever.retrieve(query)[0]
    return retrieved_node.get_content()


def build_retrieve_text_tool(context: ToolingContext) -> FunctionTool:
    return FunctionTool.from_defaults(
        fn=partial(_retrieve_text_function, context=context),
        name="retrieve_text_tool",
        description="Retrieves text from STORAGE based on a natural language query.",
        fn_schema=_RetrieveTextOutput,
    )
