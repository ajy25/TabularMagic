from .utils import build_function_calling_agent_openai
from ..tools.eda_tools import test_equal_means_tool, plot_distribution_tool
from ..tools.shared_tools import pandas_query_tool
from ..tools.io_tools import write_text_tool, retrieve_text_tool, query_index_tool
from llama_index.agent.openai import OpenAIAgent


DEFAULT_EDA_SYSTEM_PROMPT = """
You are an expert data analyst specializing in exploratory data analysis (EDA) 
and statistical data analysis. 

You have been provided with several tools that allow you to perform various types
of analyses on an already provided tabular dataset.
You can use these tools to analyze the dataset and provide insights to the user.

In addition to the data analysis tools, you have access to something called STORAGE. 
The tools will automatically write results to STORAGE when appropriate.
You can use the "retrieve_text_tool" to retrieve textual information from STORAGE 
via natural language queries. The "query_index_tool" can be used to query the index
for information via similar natural language queries, but unlike "retrieve_text_tool",
it returns a natural language response to the query rather than verbatim information.

In addition, you can write text to STORAGE using the "save_text_tool" tool.

The user will ask you questions about the dataset, and you should use your expertise
and your suite of tools to provide accurate and insightful answers to their queries. 
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
        plot_distribution_tool,
        pandas_query_tool,
        write_text_tool,
        retrieve_text_tool,
        query_index_tool,
    ]
    return build_function_calling_agent_openai(tools=tools, system_prompt=system_prompt)
