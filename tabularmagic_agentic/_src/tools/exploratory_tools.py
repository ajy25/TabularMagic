from langchain.tools import BaseTool
import tabularmagic as tm
from langchain.pydantic_v1 import BaseModel, Field
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from typing import Type, Optional, Literal






class TTest_Tool(BaseTool):
    """LangChain tool for conducting ttests.
    """






