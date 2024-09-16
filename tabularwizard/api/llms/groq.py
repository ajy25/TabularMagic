from langchain_groq import ChatGroq
from .find_key_from_dot_env import find_key


def build_chat_groq(
    model: str = "llama-3.1-70b-versatile", temperature: float = 0.2
) -> ChatGroq:
    """Builds a ChatGroq object. If no Groq API key is found in the .env file,
    raises a ValueError.

    Parameters
    ----------
    model : str
        The model to use, by default "llama-3.1-70b-versatile".

    temperature : float
        The temperature of the model, by default 0.2.

    Returns
    -------
    ChatGroq
        A ChatGroq object.
    """
    return ChatGroq(model=model, temperature=temperature, api_key=find_key("groq"))
