from typing import Literal
import textwrap
from .constants import TOSTR_MAX_WIDTH
import re



def color_text(text, color: Literal['red', 'blue', 'green', 'yellow', 
                'purple', 'none']):
    return text
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
    

def print_wrapped(text: str, 
                  type: Literal['WARNING', 'UPDATE', None] = None):
    """Prints text to terminal/console.

    Parameters
    ----------
    - text : str.
    - type : Literal['WARNING', 'UPDATE', None].
    """
    base_message = text
    indent_size='    '
    if type == 'WARNING':
        base_message = color_text('WARN: ', 'red') + base_message
        indent_size = ' '*len('WARN: ')
    elif type == 'UPDATE':
        base_message = color_text('INFO: ', 'green') + base_message
        indent_size = ' '*len('INFO: ')

    print(
        textwrap.fill(
            base_message, 
            width=TOSTR_MAX_WIDTH
        )
    )


def list_to_string(lst, color: Literal['red', 'blue', 'green', 'yellow', 
                'purple', 'none'] = 'none'):
    """
    Converts a Python list to a string representation with 
    elements separated by commas.

    Args:
        lst (list): The input list to be converted to a string.

    Returns:
        str: A string representation of the input list with elements 
        separated by commas.
    """
    # Join the string representations of each element in the list with a comma
    
    msg = ''
    for i, elem in enumerate(lst):
        if i == len(lst) - 1:
            msg += f'{color_text(elem, color)}'
        else:
            msg += f'{color_text(elem, color)}, '
    return msg




import re
import textwrap

def fill_ignore_color(text, width=70, **kwargs):
    """
    Wrap text, ignoring ANSI escape sequences when computing line length.

    :param text: The text to wrap
    :param width: The maximum width of each line
    :param kwargs: Additional keyword arguments passed to `textwrap.fill`
    :return: The wrapped text
    """
    # Remove ANSI escape sequences from the text
    ansi_escape_sequences = r'\033\[(\d+m)'
    text_no_ansi = re.sub(ansi_escape_sequences, '', text)

    # Compute the wrapped text, using the text without ANSI escape sequences
    wrapped_text = textwrap.fill(text_no_ansi, width=width, **kwargs)

    # Replace the original text (with ANSI escape sequences) into the wrapped text
    lines = wrapped_text.split('\n')
    wrapped_lines = []
    for line in lines:
        ansi_sequences = re.finditer(ansi_escape_sequences, text)
        ansi_positions = [(m.start(), m.end()) for m in ansi_sequences]
        ansi_positions.sort(reverse=True)

        line_with_ansi = ''
        last_pos = 0
        for pos, end in ansi_positions:
            line_with_ansi += text[last_pos:pos]
            line_with_ansi += text[pos:end]
            last_pos = end
        line_with_ansi += text[last_pos:]

        wrapped_lines.append(line_with_ansi)
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text
