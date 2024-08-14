from typing import Literal
import sys
import os
from contextlib import contextmanager
from .print_options import print_options


def color_text(
    text, color: Literal["red", "blue", "green", "yellow", "purple", "none"]
) -> str:
    """Returns text in a specified color.

    Parameters
    ----------
    text : str
        The input text.

    color : Literal['red', 'blue', 'green', 'yellow', 'purple', 'none']
        The color of the text.

    Returns
    -------
    str
    """
    if color == "none":
        return text
    elif color == "red":
        return "\033[91m" + text + "\033[0m"
    elif color == "blue":
        return "\033[94m" + text + "\033[0m"
    elif color == "purple":
        return "\033[95m" + text + "\033[0m"
    elif color == "green":
        return "\033[92m" + text + "\033[0m"
    elif color == "yellow":
        return "\033[93m" + text + "\033[0m"


def bold_text(text):
    """Returns text in bold."""
    return "\033[1m" + text + "\033[0m"


def print_wrapped(
    text: str,
    type: Literal["WARNING", "UPDATE", "PROGRESS", None] = None,
    level: Literal["INFO", "DEBUG"] = "INFO",
):
    """Logs text.

    Parameters
    ----------
    text : str.
    type : Literal['WARNING', 'UPDATE', 'PROGRESS', None].
        Default: None.
    level : Literal['INFO', 'DEBUG'].
        Default: 'INFO'.
    """
    base_message = text
    if type == "WARNING":
        base_message = color_text("WARN: ", "red") + base_message
    elif type == "UPDATE":
        base_message = color_text("INFO: ", "green") + base_message
    elif type == "PROGRESS":
        base_message = color_text("PROG: ", "yellow") + base_message

    if level == "DEBUG":
        print_options._log_debug(
            fill_ignore_format(base_message, width=print_options._max_line_width)
        )
    elif level == "INFO":
        print_options._log_info(
            fill_ignore_format(base_message, width=print_options._max_line_width)
        )


def list_to_string(
    lst,
    color: Literal["red", "blue", "green", "yellow", "purple", "none"] = "purple",
    include_quotes: bool = True,
) -> str:
    """
    Converts a Python list to a string representation with
    elements separated by commas.

    Parameters
    ----------
    lst : list
        The input list.

    color : Literal['red', 'blue', 'green', 'yellow', 'purple', 'none']
        Default: 'purple'. The color of the elements.

    include_quotes : bool
        Default: True. If True, the elements are enclosed in quotes.

    Returns
    -------
    str
    """
    msg = ""
    for i, elem in enumerate(lst):
        if include_quotes:
            elem = f"'{elem}'"
        if i == len(lst) - 1:
            msg += color_text(elem, color)
        else:
            msg += color_text(elem + ", ", color)
    return msg


def len_ignore_format(text: str) -> int:
    """Returns the length of a string without ANSI codes."""
    base_len = len(text)
    if "\033[91m" in text:
        count = text.count("\033[91m")
        base_len -= 5 * count
    if "\033[92m" in text:
        count = text.count("\033[92m")
        base_len -= 5 * count
    if "\033[93m" in text:
        count = text.count("\033[93m")
        base_len -= 5 * count
    if "\033[94m" in text:
        count = text.count("\033[94m")
        base_len -= 5 * count
    if "\033[95m" in text:
        count = text.count("\033[95m")
        base_len -= 5 * count
    if "\033[1m" in text:
        count = text.count("\033[1m")
        base_len -= 4 * count
    if "\033[0m" in text:
        count = text.count("\033[0m")
        base_len -= 4 * count
    return base_len


def fill_ignore_format_single_line(
    text: str,
    width: int = print_options._max_line_width,
    initial_indent: int = 0,
    subsequent_indent: int = 6,
) -> str:
    """Wraps text to a max width of TOSTR_MAX_WIDTH. Text must NOT
    contain any newline characters.

    Parameters
    ----------
    text : str
        The text to be wrapped.

    width : int
        Default: print_options._max_line_width. The maximum width of the wrapped text.

    initial_indent : int
        Default: 0. The number of spaces to indent the first line.

    subsequent_indent : int
        Default: 6. The number of spaces to indent subsequent lines.

    Returns
    -------
    str
    """
    if "\n" in text:
        raise ValueError("Text must not contain newline characters.")

    text_split = text.split(" ")
    newstr = ""

    newstr += " " * initial_indent
    line_length = initial_indent

    for word in text_split:
        if line_length + len_ignore_format(word) > width:
            newstr += "\n"
            newstr += " " * subsequent_indent
            line_length = subsequent_indent
        newstr += word + " "
        line_length += len_ignore_format(word) + 1

    return newstr


def fill_ignore_format(
    text: str,
    width: int = print_options._max_line_width,
    initial_indent: int = 0,
    subsequent_indent: int = 6,
) -> str:
    """Wraps text to a max width of TOSTR_MAX_WIDTH.

    Parameters
    ----------
    test : str
        The text to be wrapped.

    width : int
        Default: print_options._max_line_width. The maximum width of the wrapped text.

    initial_indent : int
        Default: 0. The number of spaces to indent the first line.

    subsequent_indent : int
        Default: 6. The number of spaces to indent subsequent lines.

    Returns
    -------
    str
    """
    return "\n".join(
        [
            fill_ignore_format_single_line(
                line, width, initial_indent, subsequent_indent
            )
            for line in text.split("\n")
        ]
    )


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
