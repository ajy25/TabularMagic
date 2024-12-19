problematic_chars = [
    " ",
    ".",
    "[",
    "]",
    "{",
    "}",
    "\\",
    "|",
    "&",
    "%",
    "$",
    "#",
    "\n",
    "\t",
]


def is_valid_var_name(name: str) -> bool:
    for char in problematic_chars:
        if char in name:
            return False
    return True


def rename_var(var: str) -> str:
    """Returns a cleaned version of the variable name."""
    new_var = var
    for char in problematic_chars:
        new_var = new_var.replace(char, "_")
    return new_var


def rename_vars(vars: list[str]) -> dict[str, str]:
    """Returns a list of original variable names mapped to their cleaned versions."""
    vars = vars.copy()
    output = {}
    for var in vars:
        new_var = var
        for char in problematic_chars:
            new_var = new_var.replace(char, "_")
        output[var] = new_var
    return output
