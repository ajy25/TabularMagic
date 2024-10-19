from llama_index.agent.openai import OpenAIAgent

from .utils import build_function_calling_agent_openai
from ..tools.io_tools import write_text_tool, retrieve_text_tool, query_index_tool
from ..tools.data_tools import pandas_query_tool, get_variable_description_tool
from ..tools.linear_regression_tools import ols_tool
from .system_prompts.storage_system_prompt import STORAGE_SYSTEM_PROMPT


DEFAULT_LINEAR_REGRESSION_SYSTEM_PROMPT = f"""
You are an expert data analyst specializing in linear regression.
You have been provided with several tools that allow you to perform various linear regression tasks.
You can use these tools to analyze datasets and provide insights to the user.

{STORAGE_SYSTEM_PROMPT}

The user will ask you questions about the dataset, and you should use your expertise and your suite of tools to provide accurate and insightful answers to their queries. 
Your response should contain as much detail as possible.
If you cannot answer the question with your tools, let the user know.
"""


# If you require the variable descriptions, you can use the 'get_variable_description_tool' tool to get the description of a variable.
# If a variable description is not available, you can ask the user to provide a description of the variable.


def build_linear_regression_agent(
    system_prompt: str = DEFAULT_LINEAR_REGRESSION_SYSTEM_PROMPT,
) -> OpenAIAgent:
    """Builds a linear regression agent.

    Parameters
    ----------
    system_prompt : str
        System prompt. Default linear regression system prompt is used if not provided.

    Returns
    -------
    OpenAIAgent
        OpenAI agent
    """
    tools = [
        write_text_tool,
        retrieve_text_tool,
        query_index_tool,
        pandas_query_tool,
        # get_variable_description_tool,
        ols_tool,
    ]
    return build_function_calling_agent_openai(tools=tools, system_prompt=system_prompt)
