from llama_index.core.tools import FunctionTool
from llama_index.core.llms.function_calling import FunctionCallingLLM

from pydantic import BaseModel, Field

from .eda_agent import build_eda_agent
from .linear_regression_agent import build_linear_regression_agent
from .ml_agent import build_ml_agent
from .utils import build_function_calling_agent


from ..tools.data_tools import build_pandas_query_tool

from ..tools.tooling_context import ToolingContext


ORCHESTRATOR_SYSTEM_PROMPT = f"""
You are an orchestrator agent that can interact with several data analysis agents to answer user queries.
A dataset has already been loaded into the system. Your agents and your tools will help you analyze this dataset.

Each agent has access to a suite of tools that allow them to perform various types of analyses on an already provided tabular dataset.
They can give you insights based on their tools.
Here are the agents you can interact with:

1. EDA Agent: An expert data analyst specializing in exploratory data analysis (EDA) and statistical data analysis.
    Call this agent if you need:
        - Summary statistics of the dataset.
        - Information about the distribution of specific variables.
        - To test for differences in means between groups.

2. Linear Regression Agent: An expert data analyst specializing in linear regression.
    Call this agent if you need:
        - To perform a linear regression analysis.

3. Machine Learning Agent: An expert data analyst specializing in machine learning and model comparison.
    Call this agent if you need:
        - To perform machine learning regression or classification.

To interact with an agent, simply call the tool corresponding to the correct agent. Pass along the query you received verbatim to the agent, as plain text. 

In addition to the agents, you have tools which allow you to access the STORAGE.
Your agents also have access to these storage tools, so they can store and retrieve information as needed.
Your agents will inform you when they have written something to STORAGE.
You can use the 'retrieve_text_tool' to retrieve textual information from STORAGE via natural language queries. Relevant information will be returned to you verbatim.
The STORAGE can help you summarize your analysis and prevent you from repeating the same analysis multiple times.
The user should never know about STORAGE, so you should not mention it in your responses. 
You should not mention 'save', 'store', 'retrieve', 'query', or any other related terms in your responses.

The user will ask you questions about the dataset, and you should use your expertise and your suite of tools to provide accurate and insightful answers to their queries.
Your responses should be as concise yet informative as possible.
Feel free to guess what the user might be asking for in order to use a tool. 
Avoid answering without either using a tool or checking the STORAGE.
NEVER answer with novel code—only use the tools provided to you.
"""


# You also have access to two tools that can help you get or set information about the dataset:

# 1. The 'get_variable_description_tool' tool allows you to get the description of a variable.
# 2. The 'set_variable_description_tool' tool allows you to set the description of a variable.

# It is your responsibility to use the 'set_variable_description_tool' to set the description of a variable if the user provides it.
# If one of your agents asks for a variable description, you should ask the user for the description, then use the 'set_variable_description_tool' to set the description of the variable for future reference.


class Orchestrator:
    """Class for orchestrating the interactions between the user and the LLMs."""

    def __init__(self, llm: FunctionCallingLLM, context: ToolingContext, react: bool):
        """Initializes the Orchestrator object."""

        self._eda_agent = build_eda_agent(llm=llm, context=context, react=react)
        self._linear_regression_agent = build_linear_regression_agent(
            llm=llm, context=context, react=react
        )
        self._ml_agent = build_ml_agent(llm=llm, context=context, react=react)

        class _EdaAgentTool(BaseModel):
            query: str = Field(
                description="Natural language query to ask the EDA agent."
            )

        def eda_agent_fn(query: str) -> str:
            """Use this function to interact with the EDA agent.

            Takes in a natural language query and returns the natural language response from the EDA agent.
            """
            return self._eda_agent.chat(query)

        eda_agent_tool = FunctionTool.from_defaults(
            name="eda_agent_tool",
            fn=eda_agent_fn,
            description="Call this tool to ask the EDA agent a question. "
            "The EDA agent can provide summary statistics of the dataset, "
            "information about the distribution of specific variables, "
            "and test for differences in means between groups. "
            "Pass along the query you received verbatim to the agent, as plain text. "
            "Do not ask the agent anything not related to exploratory data analysis.",
            fn_schema=_EdaAgentTool,
        )

        class _LinearRegressionAgentTool(BaseModel):
            query: str = Field(
                description="Natural language query to ask the Linear Regression agent."
            )

        def linear_regression_agent_fn(query: str) -> str:
            """Use this function to interact with the Linear Regression agent.

            Takes in a natural language query and returns the natural language response from the EDA agent.
            """
            return self._linear_regression_agent.chat(query)

        linear_regression_agent_tool = FunctionTool.from_defaults(
            name="linear_regression_agent_tool",
            fn=linear_regression_agent_fn,
            description="Call this tool to ask the Linear Regression agent a question. "
            "The Linear Regression agent can perform a linear regression analysis. "
            "Pass along the query you received verbatim to the agent, as plain text. "
            "Do not ask the agent anything not related to linear regression.",
            fn_schema=_LinearRegressionAgentTool,
        )

        class _MLAgentTool(BaseModel):
            query: str = Field(
                description="Natural language query to ask the ML agent."
            )

        def ml_agent_fn(query: str) -> str:
            """Use this function to interact with the ML agent.

            Takes in a natural language query and returns the natural language response from the ML agent.
            """
            return self._ml_agent.chat(query)

        ml_agent_tool = FunctionTool.from_defaults(
            name="ml_agent_tool",
            fn=ml_agent_fn,
            description="Call this tool to ask the ML agent a question. "
            "The ML agent can perform machine learning tasks. "
            "Pass along the query you received verbatim to the agent, as plain text. "
            "Do not ask the agent anything not related to machine learning.",
            fn_schema=_MLAgentTool,
        )

        tools = [
            eda_agent_tool,
            linear_regression_agent_tool,
            ml_agent_tool,
            build_pandas_query_tool(context),
        ]

        self.agent = build_function_calling_agent(
            llm=llm, tools=tools, system_prompt=ORCHESTRATOR_SYSTEM_PROMPT, react=react
        )

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
