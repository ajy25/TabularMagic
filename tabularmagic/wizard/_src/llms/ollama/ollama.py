from llama_index.llms.ollama import Ollama


def build_ollama(
    model: str = "llama3-groq-tool-use", temperature: float = 0.0
) -> Ollama:
    """Builds a Ollama object using LlamaIndex.

    Parameters
    ----------
    model : str
        The model to use, by default "llama3.2".
        Must support function calling.

    temperature : float
        The temperature of the model, by default 0.0.

    Returns
    -------
    Ollama
        An Ollama object from LlamaIndex.
    """
    return Ollama(model=model, temperature=temperature, request_timeout=60)


if __name__ == "__main__":
    ollama = build_ollama()
    print(ollama.complete("Hello, World!"))
