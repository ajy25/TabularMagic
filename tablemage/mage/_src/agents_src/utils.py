from llama_index.agent.openai import OpenAIAgent
from llama_index.core.agent import FunctionCallingAgent, ReActAgent
from llama_index.core.llms.function_calling import FunctionCallingLLM
from ..llms.openai.openai import build_openai


def build_function_calling_agent_openai(
    tools: list, system_prompt: str, verbose: bool = True
) -> OpenAIAgent:
    """Builds a function calling agent.

    Parameters
    ----------
    tools : list
        List of tools.

    system_prompt : str
        System prompt.

    Returns
    -------
    OpenAIAgent
        OpenAI agent
    """
    return OpenAIAgent.from_tools(
        tools=tools, llm=build_openai(), verbose=verbose, system_prompt=system_prompt
    )


def build_function_calling_agent(
    llm: FunctionCallingLLM,
    tools: list,
    system_prompt: str,
    verbose: bool = True,
    react: bool = False,
) -> FunctionCallingAgent | ReActAgent:
    """Builds a function calling agent.

    Parameters
    ----------
    llm : FunctionCallingLLM
        Language model.

    tools : list
        List of tools.

    system_prompt : str
        System prompt.

    verbose : bool
        If True, print to sys stdout.

    react : bool
        If True, build a ReAct agent instead of a function calling agent.
        Voids the system prompt if True (uses ReAct default system prompt).

    Returns
    -------
    FunctionCallingAgent | ReActAgent
        Function calling agent or ReAct agent
    """
    if react:
        return ReActAgent.from_tools(tools=tools, llm=llm, verbose=verbose)
    else:
        return FunctionCallingAgent.from_tools(
            tools=tools, llm=llm, verbose=verbose, system_prompt=system_prompt
        )
