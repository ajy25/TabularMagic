from llama_index.agent.openai import OpenAIAgent

from .utils import build_function_calling_agent_openai
from ..tools.eda_tools import (
    test_equal_means_tool,
    plot_distribution_tool,
    numeric_summary_statistics_tool,
    categorical_summary_statistics_tool,
)
from ..tools.io_tools import write_text_tool, retrieve_text_tool, query_index_tool
from ..tools.data_tools import pandas_query_tool, get_variable_description_tool
from .system_prompts.storage_system_prompt import STORAGE_SYSTEM_PROMPT


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


def build_eda_agent(system_prompt: str = EDA_SYSTEM_PROMPT) -> OpenAIAgent:
    """Builds an EDA agent.

    Parameters
    ----------
    system_prompt : str
        System prompt. Default EDA system prompt is used if not provided.

    Returns
    -------
    OpenAIAgent
        OpenAI agent
    """
    tools = [
        test_equal_means_tool,
        plot_distribution_tool,
        numeric_summary_statistics_tool,
        categorical_summary_statistics_tool,
        write_text_tool,
        retrieve_text_tool,
        query_index_tool,
        pandas_query_tool,
        # get_variable_description_tool
    ]
    return build_function_calling_agent_openai(tools=tools, system_prompt=system_prompt)
