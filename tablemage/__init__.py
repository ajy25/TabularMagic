"""
TabularMagic
------------
A low-code Python package for analyzing wide format tabular data.


"""

from ._src.analyzer import Analyzer
from . import ml
from . import options
from . import fs


__all__ = ["Analyzer", "ml", "options", "fs"]


__version__ = "0.1.0a1"
