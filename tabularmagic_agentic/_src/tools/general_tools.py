from langchain.tools import BaseTool
import tabularmagic as tm
from langchain.pydantic_v1 import BaseModel, Field
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from typing import Type, Optional, Literal


class DataOverview_Input(BaseModel):
    dataset: str = Field(
        description="Specifies which dataset to provide the overview on. "
        "Either 'train' or 'test'."
    )

class DataOverview_Tool(BaseTool):
    """LangChain tool for retrieving an overview of a TabularMagic dataset."""

    name = "Data Overview Tool"
    description = "This tool returns a string that broadly describes the dataset. "
    "The string output includes the number of examples (rows), the number of variables "
    "(columns), a list of numerical variable names, and a list of categorical "
    "variable names."
    args_schema: Type[BaseModel] = DataOverview_Input

    def __init__(self, tm_analyzer: tm.Analyzer):
        """Initializes the tool.

        Parameters
        ----------
        tm_analyzer : tm.Analyzer.
        """
        super().__init__()
        self._tm_analyzer = tm_analyzer

    def _run(
        self, 
        dataset: Literal["train", "test"], 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        output = ""

        if dataset == "train":
            eda = self._tm_analyzer.eda("train")
        elif dataset == "test":
            eda = self._tm_analyzer.eda("test")
        else:
            raise ValueError("Invalid input for parameter 'dataset'. "
                             "Must be either 'train' or 'test'. ")
        
        n_examples, n_vars = eda.df.shape
        output += f"The {dataset} dataset has {n_examples} examples (rows) and "
        f"{n_vars} variables (columns). "
        
        num_vars = ", ".join(eda.numerical_vars())
        cat_vars = ", ".join(eda.categorical_vars())
        output += f"The numerical variables are: {num_vars}. "
        output += f"The categorical variables are: {cat_vars}. "
    
    async def _arun(
        self,
        dataset: str, 
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError(f"{self.name} does not support async.")









