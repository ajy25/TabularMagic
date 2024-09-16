from langchain_openai import ChatOpenAI

from .find_key_from_dot_env import find_key


def build_chat_openai(
    model: str = "gpt-4o-mini", temperature: float = 0.2
) -> ChatOpenAI:
    """Builds a ChatOpenAI object. If no OpenAI API key is found in the .env file,
    raises a ValueError.

    Parameters
    ----------
    model : str
        The model to use.

    temperature : float
        The temperature of the model, by default 0.2.

    Returns
    -------
    ChatOpenAI
        A ChatOpenAI object.
    """
    return ChatOpenAI(model=model, temperature=temperature, api_key=find_key("openai"))
