from llama_index.agent.openai import OpenAIAgent
from ..llms.openai import build_chat_openai


def build_function_calling_agent_openai(tools: list, system_prompt: str) -> OpenAIAgent:
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
        tools=tools, llm=build_chat_openai(), verbose=True, system_prompt=system_prompt
    )
