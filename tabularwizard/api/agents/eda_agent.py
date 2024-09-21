from .utils import build_function_calling_agent_openai
from ..tools.eda_tools import test_equal_means_tool, numerical_summary_stats_tool
from ..tools.shared_tools import sql_query_tool
from llama_index.agent.openai import OpenAIAgent


DEFAULT_EDA_SYSTEM_PROMPT = """
You are an expert data analyst specializing in exploratory data analysis (EDA).
You have been provided with several tools that allow you to perform various EDA tasks.
You can use these tools to analyze datasets and provide insights to the user.
Respond to the user's queries with the appropriate analysis results.
Be concise and informative in your responses.
Use new lines frequently to make your responses more readable.
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
    tools = [test_equal_means_tool, numerical_summary_stats_tool, sql_query_tool]
    return build_function_calling_agent_openai(tools=tools, system_prompt=system_prompt)
