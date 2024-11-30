from llama_index.core.agent import FunctionCallingAgent, ReActAgent
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.core.agent import FunctionCallingAgent


from .._debug.logger import print_debug

from ..tools.ml_tools import (
    build_ml_regression_tool,
    build_ml_classification_tool,
    build_feature_selection_tool,
)
from ..tools.eda_tools import (
    build_test_equal_means_tool,
    build_plot_distribution_tool,
    build_numeric_summary_statistics_tool,
    build_categorical_summary_statistics_tool,
    build_correlation_comparison_tool,
    build_correlation_matrix_tool,
)
from ..tools.linear_regression_tools import build_ols_tool, build_logit_tool
from ..tools.data_tools import build_dataset_summary_tool, build_pandas_query_tool
from ..tools.tooling_context import ToolingContext

from .prompt.single_agent_system_prompt import SINGLE_SYSTEM_PROMPT


def build_agent(
    llm: FunctionCallingLLM,
    context: ToolingContext,
    system_prompt: str = SINGLE_SYSTEM_PROMPT,
    react: bool = False,
) -> FunctionCallingAgent | ReActAgent:
    """Builds an agent.

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
        build_feature_selection_tool(context),
        build_ml_regression_tool(context),
        build_ml_classification_tool(context),
        build_test_equal_means_tool(context),
        build_plot_distribution_tool(context),
        build_numeric_summary_statistics_tool(context),
        build_categorical_summary_statistics_tool(context),
        build_correlation_comparison_tool(context),
        build_correlation_matrix_tool(context),
        build_ols_tool(context),
        build_logit_tool(context),
        build_dataset_summary_tool(context),
        build_pandas_query_tool(context),
    ]
    obj_index = ObjectIndex.from_objects(
        tools,
        index_cls=VectorStoreIndex,
    )
    tool_retriever = obj_index.as_retriever(similarity_top_k=3)

    if react:
        agent = ReActAgent.from_tools(
            llm=llm,
            tool_retriever=tool_retriever,
            verbose=True,
            system_prompt=system_prompt,
        )
    else:
        agent = FunctionCallingAgent.from_tools(
            llm=llm,
            tool_retriever=tool_retriever,
            verbose=True,
            system_prompt=system_prompt,
        )
    return agent


class SingleAgent:

    def __init__(self, llm: FunctionCallingLLM, context: ToolingContext, react: bool):
        """Initializes the SingleAgent object."""
        if not isinstance(llm, FunctionCallingLLM):
            raise ValueError("The provided LLM must be a FunctionCallingLLM.")

        print_debug("Initializing SingleAgent")

        self._agent = build_agent(llm=llm, context=context, react=react)

        print_debug("SingleAgent initialized")

    def chat(self, message: str) -> str:
        """Interacts with the LLM to provide data analysis insights.

        Parameters
        ----------
        message : str
            The message to interact with the LLM.

        Returns
        -------
        str
            The response from the LLM.
        """
        return str(self._agent.chat(message))
