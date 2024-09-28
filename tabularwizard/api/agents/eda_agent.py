from .utils import build_function_calling_agent_openai
from ..tools.eda_tools import (
    test_equal_means_tool,
    numerical_summary_stats_tool,
    categorical_summary_stats_tool,
)
from ..tools.shared_tools import pandas_query_tool
from ..tools.io_tools import write_scratch_tool, query_scratch_tool
from llama_index.agent.openai import OpenAIAgent


DEFAULT_EDA_SYSTEM_PROMPT = """
You are an expert data analyst specializing in exploratory data analysis (EDA).
You have been provided with several tools that allow you to perform various EDA tasks.
You can use these tools to analyze datasets and provide insights to the user.
Respond to the user's queries with the appropriate analysis results.
Be concise and informative in your responses.
Use new lines frequently to make your responses more readable.

Before every response, please use the `write_scratch_tool` to 
jot down your response in the scratchpad. 
This will help you remember the context of the conversation.
Then, return the response to the user.

If needed, you can use the `query_scratch_tool` to search through your scratchpad 
using natural language queries.
"""


def build_eda_agent(system_prompt: str = DEFAULT_EDA_SYSTEM_PROMPT) -> OpenAIAgent:
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
        numerical_summary_stats_tool,
        categorical_summary_stats_tool,
        pandas_query_tool,
        write_scratch_tool,
        query_scratch_tool,
    ]
    return build_function_calling_agent_openai(tools=tools, system_prompt=system_prompt)
