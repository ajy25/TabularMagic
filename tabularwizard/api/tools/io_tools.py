from llama_index.core.tools import FunctionTool
from llama_index.core.schema import BaseNode
from pydantic import BaseModel, Field
from ..io.global_io import GLOBAL_IO


# save text tool
class _WriteTextInput(BaseModel):
    text: str = Field(description="The text to write to STORAGE.")
def _write_text_function(text: str) -> str:
    """Writes a text to STORAGE. Then, returns the text."""
    GLOBAL_IO.add_str(text)
    return text
write_text_tool = FunctionTool.from_defaults(
    fn=_write_text_function,
    name="write_text_tool",
    description="Writes text to STORAGE. Then, returns the text back to you. "
    "This tool is useful for storing text data for later reference.",
    fn_schema=_WriteTextInput,
)


# retrieve text tool
class _RetrieveTextOutput(BaseModel):
    query: str = Field(
        description="Natural language query to retrieve information in STORAGE. "
        "For example, if STORAGE contains summary statistics, an appropriate query "
        "could be 'What is the mean of the variable \"mpg\"?'."
    )
def _retrieve_text_function(query: str) -> str:
    """Retrieves text from STORAGE based on a query."""
    retrieved_node: BaseNode = GLOBAL_IO.get_retriever().retrieve(query)[0]
    return retrieved_node.get_content()
retrieve_text_tool = FunctionTool.from_defaults(
    fn=_retrieve_text_function,
    name="retrieve_text_tool",
    description="Retrieves text from STORAGE based on a natural language query. "
    "This tool is useful for retrieving information that you have stored, "
    "either manually using 'write_text_tool' or automatically via other tool calls.",
    fn_schema=_RetrieveTextOutput,
)


# query index tool
class _QueryIndexInput(BaseModel):
    query: str = Field(
        description="Natural language query to query the STORAGE. "
        "For example, if STORAGE contains summary statistics, an appropriate query "
        "could be 'What is the mean of the variable \"mpg\"?'. "
    )
def _query_index_function(query: str) -> str:
    """Queries the index for information based on a query. """
    return str(GLOBAL_IO.query_engine.query(query))
query_index_tool = FunctionTool.from_defaults(
    fn=_query_index_function,
    name="query_index_tool",
    description="Queries the STORAGE based on a natural language query. "
    "Unlike 'retrieve_text_tool', this tool does not return verbatim information "
    "from STORAGE, but rather returns a natural language response to the query.",
    fn_schema=_QueryIndexInput,
)


