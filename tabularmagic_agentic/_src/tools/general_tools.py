from langchain.tools import tool
import tabularmagic as tm
from langchain.pydantic_v1 import BaseModel, Field
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from typing import Type, Optional, Literal
from ..shared_analyzer import shared_analyzer


class _DataOverview_ToolArgs(BaseModel):
    dataset: str = Field(
        description="Specifies which dataset to provide the overview on. "
        "Either 'train' or 'test'."
    )


@tool("data_overview_tool", args_schema=_DataOverview_ToolArgs)
def data_overview_tool(dataset: Literal["train", "test"]) -> str:
    """Returns variable information for the dataset in as a json-formatted string."""
    return (
        shared_analyzer.get_shared_analyzer()
        .eda(dataset=dataset)
        ._agentic_describe_json_str()
    )
