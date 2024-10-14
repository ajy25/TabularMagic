"""
TabularMagic
------------
A low-code Python package for analyzing wide format tabular data.


"""

from ._src.analyzer import Analyzer
from . import ml
from . import options
from . import fs

# try:
#     from . import wizard
# except ImportError:
#     pass

__all__ = ["Analyzer", "ml", "options", "fs"]


__version__ = "0.1.0a1"
