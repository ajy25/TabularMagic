from llama_index.agent.openai import OpenAIAgent

from .utils import build_function_calling_agent_openai
from ..tools.eda_tools import (
    build_test_equal_means_tool,
    build_plot_distribution_tool,
    build_numeric_summary_statistics_tool,
    build_categorical_summary_statistics_tool,
)
from ..tools.io_tools import build_write_text_tool, build_retrieve_text_tool
from ..tools.data_tools import build_pandas_query_tool
from .system_prompts.storage_system_prompt import STORAGE_SYSTEM_PROMPT
from ..tools.tooling_context import ToolingContext


EDA_SYSTEM_PROMPT = f"""
You are an expert data analyst specializing in exploratory data analysis (EDA) and statistical data analysis. 
You have been provided with several tools that allow you to perform various types of analyses on an already provided tabular dataset.
You can use these tools to analyze the dataset and provide insights to the user.

{STORAGE_SYSTEM_PROMPT}

The user will ask you questions about the dataset, and you should use your expertise and your suite of tools to provide accurate and insightful answers to their queries. 
Your response should contain as much detail as possible.
If you cannot answer the question with your tools, let the user know.
"""

# If you require the variable descriptions, you can use the 'get_variable_description_tool' tool to get the description of a variable.
# If a variable description is not available, you can ask the user to provide a description of the variable.


def build_eda_agent(
    context: ToolingContext, system_prompt: str = EDA_SYSTEM_PROMPT
) -> OpenAIAgent:
    """Builds an EDA agent.

    Parameters
    ----------
    context : ToolingContext
        Tooling context

    system_prompt : str
        System prompt. Default EDA system prompt is used if not provided.

    Returns
    -------
    OpenAIAgent
        OpenAI agent
    """
    tools = [
        build_test_equal_means_tool(context),
        build_plot_distribution_tool(context),
        build_numeric_summary_statistics_tool(context),
        build_categorical_summary_statistics_tool(context),
        build_write_text_tool(context),
        build_retrieve_text_tool(context),
        build_pandas_query_tool(context),
    ]
    return build_function_calling_agent_openai(tools=tools, system_prompt=system_prompt)
