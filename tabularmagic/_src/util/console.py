from typing import Literal
from .constants import TOSTR_MAX_WIDTH



def color_text(text, color: Literal['red', 'blue', 'green', 'yellow', 
                'purple', 'none']):
    """Returns text in a specified color."""
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
    

def bold_text(text):
    """Returns text in bold.
    """
    return '\033[1m' + text + '\033[0m'
    

def print_wrapped(text: str, 
                  type: Literal['WARNING', 'UPDATE', None] = None):
    """Prints text to terminal/console.

    Parameters
    ----------
    - text : str.
    - type : Literal['WARNING', 'UPDATE', None].
    """
    base_message = text
    if type == 'WARNING':
        base_message = color_text('WARN: ', 'red') + base_message
    elif type == 'UPDATE':
        base_message = color_text('INFO: ', 'green') + base_message

    print(
        fill_ignore_format(
            base_message, 
            width=TOSTR_MAX_WIDTH
        )
    )





def list_to_string(lst, color: Literal['red', 'blue', 'green', 'yellow', 
                'purple', 'none'] = 'purple') -> str:
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
        elem = f'\'{elem}\''
        if i == len(lst) - 1:
            msg += color_text(elem, color)
        else:
            msg += color_text(elem + ', ', color)
    return msg



def len_ignore_format(text: str):
    """Returns the length of a string without ANSI codes."""
    base_len = len(text)
    if '\033[91m' in text:
        count = text.count('\033[91m')
        base_len -= 5 * count
    if '\033[92m' in text:
        count = text.count('\033[92m')
        base_len -= 5 * count
    if '\033[93m' in text:
        count = text.count('\033[93m')
        base_len -= 5 * count
    if '\033[94m' in text:
        count = text.count('\033[94m')
        base_len -= 5 * count
    if '\033[95m' in text:
        count = text.count('\033[95m')
        base_len -= 5 * count
    if '\033[1m' in text:
        count = text.count('\033[1m')
        base_len -= 4 * count
    if '\033[0m' in text:
        count = text.count('\033[0m')
        base_len -= 4 * count
    return base_len





def fill_ignore_format_single_line(text: str, width: int = TOSTR_MAX_WIDTH, 
        initial_indent: int = 0, subsequent_indent: int = 6):
    """Wraps text to a max width of TOSTR_MAX_WIDTH. Text must NOT 
    contain any newline characters.

    Parameters
    ----------
    - text : str.
    """
    if '\n' in text:
        raise ValueError('Text must not contain newline characters.')

    text_split = text.split(' ')
    newstr = ''

    newstr += ' ' * initial_indent
    line_length = initial_indent

    for word in text_split:
        if line_length + len_ignore_format(word) > width:
            newstr += '\n'
            newstr += ' ' * subsequent_indent
            line_length = subsequent_indent
        newstr += word + ' '
        line_length += len_ignore_format(word) + 1

    return newstr





def fill_ignore_format(text: str, width: int = TOSTR_MAX_WIDTH, 
                      initial_indent: int = 0,
                      subsequent_indent: int = 6):
    """Wraps text to a max width of TOSTR_MAX_WIDTH.

    Parameters
    ----------
    - text : str.
    """
    return '\n'.join(
        [fill_ignore_format_single_line(line, width, 
            initial_indent, subsequent_indent) for line in text.split('\n')]
    )



