from typing import Literal, Callable

from ._debug.logger import print_debug
from .llms.groq.groq import build_groq
from .llms.openai.openai import build_openai, build_openai_multimodal
from .llms.ollama.ollama import build_ollama
from .llms.togetherai.togetherai import build_togetherai
from .llms.api_key_utils import key_exists


class _WizardOptions:

    def __init__(self):
        self._multimodal_llm_build_function = None
        self._llm_build_function = None

        if key_exists("openai"):
            self._llm_build_function = build_openai
            self._multimodal_llm_build_function = build_openai_multimodal
        elif key_exists("groq"):
            self._llm_build_function = build_groq
        else:
            raise ValueError("No API keys found in .env file.")

        self._multimodal = (
            True if self._multimodal_llm_build_function is not None else False
        )
        print_debug(
            "A new _WizardOptions object has been created "
            "with LLM build function: "
            + str(self._llm_build_function)
            + " and multimodal LLM build function: "
            + str(self._multimodal_llm_build_function)
        )

    def set_llm(
        self, llm_type: Literal["openai", "groq", "ollama", "togetherai"]
    ) -> None:
        """Sets the LLM type.

        Parameters
        ----------
        llm_type : Literal["openai", "groq"]
            The type of LLM to use.
        """
        if llm_type == "openai":
            if not key_exists("openai"):
                raise ValueError("OpenAI API key not found in .env file.")
            self._llm_build_function = build_openai
        elif llm_type == "groq":
            if not key_exists("groq"):
                raise ValueError("GROQ API key not found in .env file.")
            self._llm_build_function = build_groq
        elif llm_type == "ollama":
            self._llm_build_function = build_ollama
        elif llm_type == "togetherai":
            if not key_exists("togetherai"):
                raise ValueError("TogetherAI API key not found in .env file.")
            self._llm_build_function = build_togetherai
        else:
            raise ValueError("Invalid LLM type specified.")

    @property
    def llm_build_function(self) -> Callable:
        """The function to build the LLM."""
        return self._llm_build_function

    @property
    def multimodal_llm_build_function(self) -> Callable:
        """The function to build the multimodal LLM."""
        return self._multimodal_llm_build_function

    @property
    def multimodal(self) -> bool:
        """Whether the LLM is multimodal."""
        return self._multimodal


options = _WizardOptions()
