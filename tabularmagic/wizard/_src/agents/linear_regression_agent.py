from llama_index.core.agent import FunctionCallingAgent, ReActAgent
from llama_index.core.llms.function_calling import FunctionCallingLLM

from .utils import build_function_calling_agent
from ..tools.linear_regression_tools import build_ols_tool, build_logit_tool
from ..tools.tooling_context import ToolingContext
from .prompt.linear_regression_agent_system_prompt import (
    LINEAR_REGRESSION_SYSTEM_PROMPT,
)


def build_linear_regression_agent(
    llm: FunctionCallingLLM,
    context: ToolingContext,
    system_prompt: str = LINEAR_REGRESSION_SYSTEM_PROMPT,
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
        # build_write_text_tool(context),
        build_ols_tool(context),
        build_logit_tool(context),
    ]
    return build_function_calling_agent(
        llm=llm, tools=tools, system_prompt=system_prompt, react=react
    )
