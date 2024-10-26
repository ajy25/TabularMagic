from llama_index.llms.groq import Groq


from ..find_key_from_dot_env import find_key


def build_groq(
    model: str = "llama3-groq-70b-8192-tool-use-preview", temperature: float = 0.0
) -> Groq:
    """Builds a OpenAI object using LlamaIndex.
    If no OpenAI API key is found in the .env file, raises a ValueError.

    Parameters
    ----------
    model : str
        The model to use, by default "llama3-groq-70b-8192-tool-use-preview".
        Must support function calling.

    temperature : float
        The temperature of the model, by default 0.0.

    Returns
    -------
    OpenAI
        An OpenAI object from LlamaIndex.
    """
    api_key = find_key("groq")
    if not api_key:
        raise ValueError("No Groq API key found in .env file")

    return Groq(model=model, temperature=temperature, api_key=api_key)
