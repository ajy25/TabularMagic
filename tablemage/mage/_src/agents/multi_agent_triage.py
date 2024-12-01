from llama_index.core.llms.function_calling import FunctionCallingLLM

from .eda_agent import build_eda_agent
from .linear_regression_agent import build_linear_regression_agent
from .ml_agent import build_ml_agent
from ..tools.tooling_context import ToolingContext
from .._debug.logger import print_debug
from .prompt.assignment_prompt import ASSIGNMENT_PROMPT


class MultiAgentAdmin:
    """Manages multiple agents. Each agent is responsible for a different task.
    Given an input, the MultiAgent will route the input to the correct agent.
    """

    def __init__(
        self, llm: FunctionCallingLLM, context: ToolingContext, react: bool = False
    ):
        if not isinstance(llm, FunctionCallingLLM):
            raise ValueError("The provided LLM must be a FunctionCallingLLM.")

        print_debug("Building EDA agent")
        self._eda_agent = build_eda_agent(llm=llm, context=context, react=react)

        print_debug("Building Linear Regression agent")
        self._linear_regression_agent = build_linear_regression_agent(
            llm=llm, context=context, react=react
        )

        print_debug("Building ML agent")
        self._ml_agent = build_ml_agent(llm=llm, context=context, react=react)

        self._llm = llm

    def chat(self, message: str) -> str:
        """Routes the message to the correct agent and returns the response."""

        # ask the llm to pick the correct task
        response = self._llm.chat(ASSIGNMENT_PROMPT.format(task=message))

        if response == "eda":
            return self._eda_agent.chat(message)
        elif response == "linear":
            return self._linear_regression_agent.chat(message)
        elif response == "ml":
            return self._ml_agent.chat(message)
