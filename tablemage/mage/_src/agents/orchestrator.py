from llama_index.core.tools import FunctionTool
from llama_index.core.llms.function_calling import FunctionCallingLLM

from pydantic import BaseModel, Field

from .eda_agent import build_eda_agent
from .linear_regression_agent import build_linear_regression_agent
from .ml_agent import build_ml_agent
from .utils import build_function_calling_agent


from ..tools.data_tools import build_dataset_summary_tool


from ..tools.tooling_context import ToolingContext
from .prompt.orchestrator_system_prompt import ORCHESTRATOR_SYSTEM_PROMPT

from .._debug.logger import print_debug


class Orchestrator:
    """Class for orchestrating the interactions between the user and the LLMs."""

    def __init__(self, llm: FunctionCallingLLM, context: ToolingContext, react: bool):
        """Initializes the Orchestrator object."""
        if not isinstance(llm, FunctionCallingLLM):
            raise ValueError("The provided LLM must be a FunctionCallingLLM.")

        print_debug("Building EDA agent")
        self._eda_agent = build_eda_agent(llm=llm, context=context, react=False)

        print_debug("Building Linear Regression agent")
        self._linear_regression_agent = build_linear_regression_agent(
            llm=llm, context=context, react=False
        )

        print_debug("Building ML agent")
        self._ml_agent = build_ml_agent(llm=llm, context=context, react=False)

        class _EdaAgentTool(BaseModel):
            query: str = Field(
                description="Natural language query for EDA agent. "
                "Use proper variable names if possible."
            )

        def eda_agent_fn(query: str) -> str:
            return self._eda_agent.chat(query)

        eda_agent_tool = FunctionTool.from_defaults(
            name="eda_agent_tool",
            fn=eda_agent_fn,
            description="""Use this tool to give the EDA agent a task.
            The EDA agent can:
            - Provide summary statistics of the dataset
            - Provide information about the distribution of variables
            - Provide information about the relationship between variables, including:
                - Correlation comparisons or correlation matrix
                - Differences in means between groups
            """,
            fn_schema=_EdaAgentTool,
        )

        class _LinearRegressionAgentTool(BaseModel):
            query: str = Field(
                description="Natural language query for Linear Regression agent. "
                "Use proper variable names."
            )

        def linear_regression_agent_fn(query: str) -> str:
            return self._linear_regression_agent.chat(query)

        linear_regression_agent_tool = FunctionTool.from_defaults(
            name="linear_regression_agent_tool",
            fn=linear_regression_agent_fn,
            description="""Use this tool to give the Linear Regression agent a task.
            The Linear Regression agent can:
            - Perform linear regression (OLS)
            """,
            fn_schema=_LinearRegressionAgentTool,
        )

        class _MLAgentTool(BaseModel):
            query: str = Field(
                description="Natural language query for ML agent. "
                "Use proper variable names."
            )

        def ml_agent_fn(query: str) -> str:
            return self._ml_agent.chat(query)

        ml_agent_tool = FunctionTool.from_defaults(
            name="ml_agent_tool",
            fn=ml_agent_fn,
            description="""Use this tool to give the ML agent a task.
            The ML agent can:
            - Perform machine learning regression and classification
            """,
            fn_schema=_MLAgentTool,
        )

        tools = [
            eda_agent_tool,
            linear_regression_agent_tool,
            ml_agent_tool,
            build_dataset_summary_tool(context),
        ]

        print_debug("Initializing Orchestrator agent...")

        self.agent = build_function_calling_agent(
            llm=llm, tools=tools, system_prompt=ORCHESTRATOR_SYSTEM_PROMPT, react=react
        )

        print_debug("Orchestrator agent initialized.")

    def chat(self, message: str) -> str:
        """Interacts with the LLM to provide data analysis insights.

        Parameters
        ----------
        message : str
            The message to interact with the LLM.

        Returns
        -------
        str
            The response from the LLM.
        """
        return str(self.agent.chat(message))
