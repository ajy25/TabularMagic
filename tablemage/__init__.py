"""
TableMage
---------
Python package for low-code data science on tabular data.
"""

from ._src.analyzer import Analyzer
from . import ml
from . import options
from . import fs
from . import mage


__all__ = ["Analyzer", "ml", "options", "fs", "mage"]


__version__ = "0.1.0a1"


__author__ = "Andrew J. Yang"

