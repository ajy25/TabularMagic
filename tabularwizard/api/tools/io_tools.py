from llama_index.core.tools import FunctionTool
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from pydantic import BaseModel, Field

from ..io.jsonutils import json_cache_tracker
from ..io.scratchutils import scratch_tracker


# read json tool
class ReadJsonInput(BaseModel):
    name: str = Field(
        description="The name of the JSON file to read. "
        "The name must be in the form of 'file_\{integer\}.json'. "
        "For example, 'file_1.json'."
    )


def read_json_function(name) -> str:
    """Reads a JSON file and returns its contents."""
    return json_cache_tracker.read_json(name)


# write to scratch tool
class WriteScratchInput(BaseModel):
    scratch: str = Field(description="The scratch to write to a scratchpad.")


def write_scratch_function(scratch: str) -> str:
    """Writes a scratch to a scratchpad."""
    scratch_tracker.write_scratch(scratch)
    return "1"


write_scratch_tool = FunctionTool.from_defaults(
    fn=write_scratch_function,
    name="write_scratch_tool",
    description="Writes information to a scratchpad. The scratch can later be queried "
    "by the 'query_scratch_tool'. This tool returns '1' after writing the scratch.",
    fn_schema=WriteScratchInput,
)


# query scratch tool
class QueryScratchInput(BaseModel):
    query: str = Field(description="Natural language query to search the scratchpad.")


def query_scratch_function(query: str) -> str:
    """Queries the scratchpad."""
    documents = SimpleDirectoryReader(
        input_files=[scratch_tracker._filepath]
    ).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    return str(query_engine.query(query).response)


query_scratch_tool = FunctionTool.from_defaults(
    fn=query_scratch_function,
    name="query_scratch_tool",
    description="Queries the scratchpad using a natural language query.",
    fn_schema=QueryScratchInput,
)
