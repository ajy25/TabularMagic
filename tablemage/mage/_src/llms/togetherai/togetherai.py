from llama_index.llms.together import TogetherLLM

from ..api_key_utils import find_key


def build_togetherai(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    temperature=0.0,
) -> TogetherLLM:
    """Builds a TogetherAI object using LlamaIndex.
    If no TogetherAI API key is found in the .env file, raises a ValueError.

    Parameters
    ----------
    model : str
        The model to use, by default "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo".

    temperature : float
        The temperature of the model, by default 0.0.

    Returns
    -------
    TogetherLLM
        A TogetherAI object from LlamaIndex.
    """
    api_key = find_key("togetherai")
    if not api_key:
        raise ValueError("No TogetherAI API key found in .env file")

    return TogetherLLM(
        model=model,
        temperature=temperature,
        api_key=api_key,
        is_chat_model=True,
        is_function_calling_model=True,
    )
