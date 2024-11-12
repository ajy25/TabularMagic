from llama_index.core.agent import FunctionCallingAgent, ReActAgent
from llama_index.core.llms.function_calling import FunctionCallingLLM

from .utils import build_function_calling_agent
from ..tools.eda_tools import (
    build_test_equal_means_tool,
    build_plot_distribution_tool,
    build_numeric_summary_statistics_tool,
    build_categorical_summary_statistics_tool,
    build_correlation_comparison_tool,
    build_correlation_matrix_tool,
)
from ..tools.io_tools import build_write_text_tool
from ..tools.data_tools import build_pandas_query_tool
from ..tools.tooling_context import ToolingContext

from .system_prompts.eda_agent_system_prompt import EDA_SYSTEM_PROMPT


def build_eda_agent(
    llm: FunctionCallingLLM,
    context: ToolingContext,
    system_prompt: str = EDA_SYSTEM_PROMPT,
    react: bool = False,
) -> FunctionCallingAgent | ReActAgent:
    """Builds an EDA agent.

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
        build_test_equal_means_tool(context),
        build_plot_distribution_tool(context),
        build_numeric_summary_statistics_tool(context),
        build_categorical_summary_statistics_tool(context),
        # build_write_text_tool(context),
        build_pandas_query_tool(context),
        build_correlation_comparison_tool(context),
        build_correlation_matrix_tool(context),
    ]
    return build_function_calling_agent(
        llm=llm,
        tools=tools,
        system_prompt=system_prompt,
        react=react,
    )
