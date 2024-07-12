
import os

class _EnvVars:
    def __init__(self):
        self._openai_key = os.getenv("OPENAI_API_KEY")
    def set_openai_key(self, key: str):
        self._openai_key = key
    def get_openai_key(self) -> str | None:
        return self._openai_key


env_vars = _EnvVars()

