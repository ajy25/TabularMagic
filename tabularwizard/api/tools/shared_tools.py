from llama_index.core.tools import FunctionTool
from llama_index.experimental.query_engine import PandasQueryEngine
from pydantic import BaseModel, Field

from ..datacontainer import GLOBAL_DATA_CONTAINER
from ..llms.openai import build_openai


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
