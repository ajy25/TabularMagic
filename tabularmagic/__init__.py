"""
TabularMagic
------------
A package for analyzing wide format tabular data.

Provides:

1. Low-code interface for exploratory data analysis, regression analysis, and 
machine learning.



Exposed Classes
---------------
Analyzer
    Give an Analyzer object a DataFrame, get an easy-to-use Python 
    interface for exploring the DataFrame.


Submodules
----------
- ml
- fs
- options


Module Summary
--------------



"""

from ._src.analyzer import Analyzer
from . import ml
from . import options
from . import fs


__all__ = ["Analyzer", "ml", "options", "fs"]
