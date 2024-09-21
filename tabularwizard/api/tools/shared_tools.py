from llama_index.core.tools import FunctionTool
from llama_index.core.query_engine import NLSQLTableQueryEngine
from pydantic import BaseModel, Field

from ..tabularmagic_utils import shared_container
from ..llms.openai import build_chat_openai


# SQL query tool
class SQLQueryInput(BaseModel):
    query: str = Field(
        description="Natural language SQL query to extract information from "
        "the dataset. The query should be in plain English. "
    )


def sql_query_function(query: str) -> str:
    """Executes a natural language SQL query on the user-provided DataFrame.
    The output is also in natural language.
    """
    query_engine = NLSQLTableQueryEngine(
        sql_database=shared_container.sqldb,
        llm=build_chat_openai(),
        tables=["User-provided DataFrame"],
    )
    response = query_engine.query(query)
    return str(response)


sql_query_tool = FunctionTool.from_defaults(
    fn=sql_query_function,
    name="sql_query_tool",
    description="Executes a natural language SQL query on the user-provided DataFrame. "
    "Returns the response in natural language.",
    fn_schema=SQLQueryInput,
)
