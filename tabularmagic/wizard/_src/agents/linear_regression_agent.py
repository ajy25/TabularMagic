from llama_index.core.agent import FunctionCallingAgent, ReActAgent
from llama_index.core.llms.function_calling import FunctionCallingLLM

from .utils import build_function_calling_agent
from ..tools.io_tools import build_write_text_tool, build_retrieve_text_tool
from ..tools.data_tools import build_pandas_query_tool
from ..tools.linear_regression_tools import build_ols_tool
from .system_prompts.storage_system_prompt import STORAGE_SYSTEM_PROMPT
from ..tools.tooling_context import ToolingContext


DEFAULT_LINEAR_REGRESSION_SYSTEM_PROMPT = f"""
You are an expert data analyst specializing in linear regression.
You have been provided with several tools that allow you to perform various linear regression tasks.
You can use these tools to analyze the dataset and provide insights to the user.

{STORAGE_SYSTEM_PROMPT}

The user will ask you questions about the dataset, and you should use your expertise 
and your suite of tools to provide accurate and insightful answers to their queries. 
Your response should contain as much detail as possible.
If you cannot answer the question with your tools, let the user know.
"""


# If you require the variable descriptions, you can use the 'get_variable_description_tool' tool to get the description of a variable.
# If a variable description is not available, you can ask the user to provide a description of the variable.


def build_linear_regression_agent(
    llm: FunctionCallingLLM,
    context: ToolingContext,
    system_prompt: str = DEFAULT_LINEAR_REGRESSION_SYSTEM_PROMPT,
    react: bool = False,
) -> FunctionCallingAgent | ReActAgent:
    """Builds a linear regression agent.

    Parameters
    ----------
    llm : FunctionCallingLLM
        Function calling LLM

    context : ToolingContext
        Tooling context

    system_prompt : str
        System prompt. Default linear regression system prompt is used if not provided.

    react : bool
        If True, a ReActAgent is returned. Otherwise, a FunctionCallingAgent is returned.
        If True, the system prompt is not considered.

    Returns
    -------
    FunctionCallingAgent | ReActAgent
        Either a FunctionCallingAgent or a ReActAgent
    """
    tools = [
        build_write_text_tool(context),
        build_retrieve_text_tool(context),
        build_ols_tool(context),
        build_pandas_query_tool(context),
    ]
    return build_function_calling_agent(
        llm=llm, tools=tools, system_prompt=system_prompt, react=react
    )
