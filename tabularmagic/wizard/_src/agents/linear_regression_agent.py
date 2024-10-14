from .utils import build_function_calling_agent_openai
from ..tools.linear_regression_tools import ols_regression_tool
from ..tools.shared_tools import pandas_query_tool
from llama_index.agent.openai import OpenAIAgent


DEFAULT_LINEAR_REGRESSION_SYSTEM_PROMPT = """
You are an expert data analyst specializing in linear regression.
You have been provided with several tools that allow you to perform various 
linear regression tasks.
You can use these tools to analyze datasets and provide insights to the user.
Respond to the user's queries with the appropriate analysis results.
Be concise and informative in your responses.
Use new lines frequently to make your responses more readable.

You will analyze one dataset at a time. The dataset has already been 
loaded, and your tools can work with it directly.
"""


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
    tools = [ols_regression_tool, pandas_query_tool]
    return build_function_calling_agent_openai(tools=tools, system_prompt=system_prompt)
