from typing import Literal

def accepts_only_four(x: Literal[4]) -> None:
    return "test"

accepts_only_four(4)   # OK
accepts_only_four(19)  # Rejected