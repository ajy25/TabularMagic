from dotenv import load_dotenv
import pathlib
import os
from typing import Literal


def find_key(llm_type: Literal["openai"]) -> str:
    """Reads the .env file and returns the API key for the specified LLM type.
    If the API key is not found, raises a ValueError.

    Parameters
    ----------
    llm_type : Literal["openai"]
        The type of LLM for which to find the API key.
    """
    load_dotenv(
        dotenv_path=pathlib.Path(__file__).parent.parent.parent.parent.parent / ".env"
    )

    if llm_type == "openai":
        api_key = (
            str(os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
        )
        if api_key == "..." or api_key is None:
            raise ValueError("OpenAI API key not found in .env file.")
    else:
        raise ValueError("Invalid LLM type specified.")

    return api_key
