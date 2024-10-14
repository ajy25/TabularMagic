from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
from ..io.io import GLOBAL_IO


# save text tool
class _SaveTextInput(BaseModel):
    text: str = Field(description="The text to write to STORAGE.")


def _save_text_function(text: str) -> str:
    """Writes a text to STORAGE. Then, returns the text."""
    GLOBAL_IO.add_str(text)
    return text


save_text_tool = FunctionTool.from_defaults(
    fn=_save_text_function,
    name="save_text_tool",
    description="Writes text to STORAGE. Then, returns the text back to you. "
    "This tool is useful for storing text data for later reference.",
    fn_schema=_SaveTextInput,
)


# retrieve text tool
class _RetrieveTextOutput(BaseModel):
    query: str = Field(description="Query to search for information in STORAGE.")


def _retrieve_text_function(query: str) -> str:
    """Retrieves text from STORAGE based on a query."""
    return GLOBAL_IO.as_retriever().retrieve(query)


retrieve_text_tool = FunctionTool.from_defaults(
    fn=_retrieve_text_function,
    name="retrieve_text_tool",
    description="Retrieves text from STORAGE based on a query. "
    "This tool is useful for retrieving information that you have stored.",
    fn_schema=_RetrieveTextOutput,
)
