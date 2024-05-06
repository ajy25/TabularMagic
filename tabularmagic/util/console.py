from typing import Literal
from textwrap import fill
from .constants import TOSTR_MAX_WIDTH


def print_wrapped(text: str):
    """Prints text to terminal

    Parameters
    ----------
    - text : str
    """
    print(
        fill(
            text, width=TOSTR_MAX_WIDTH, subsequent_indent='    ', tabsize=0
        )
    )





def color_text(text, color: Literal['red', 'blue', 'green', 'yellow', 
                'purple', 'none']):
    if color == 'none':
        return text
    elif color == 'red':
        return '\033[91m' + text + '\033[0m'
    elif color == 'blue':
        return '\033[94m' + text + '\033[0m'
    elif color == 'purple':
        return '\033[95m' + text + '\033[0m'
    elif color == 'green':
        return '\033[92m' + text + '\033[0m'
    elif color == 'yellow':
        return '\033[93m' + text + '\033[0m'
    
