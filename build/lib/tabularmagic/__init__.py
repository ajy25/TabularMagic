"""
TabularMagic
------------
A package for wide format tabular data.

Provides:
1. Low-code interface for exploratory data analysis and model benchmarking
2. Automatic report generation capabilities


Exposed Classes
---------------
Analyzer.
    Give a Analyzer object a DataFrame, get an easy-to-use Python 
    interface for exploring the DataFrame.
DataHandler.
    The data-handling class. 


Submodules
----------
- ml
- fs
- options


Module Summary
--------
TabularMagic is a Python package for rapid exploratory statistical and machine 
learning modeling of wide format tabular data. TabularMagic empowers users to 
quickly explore new datasets, conduct regression analyses with ease, and 
effortlessly compute baseline performance metrics across a wide range of 
popular machine learning models. TabularMagic excels in handling datasets 
with fewer than 10,000 examples. 
"""

from ._src.analyzer import Analyzer
from ._src.data.datahandler import DataHandler
from . import ml
from . import options
from . import fs


__all__ = ["Analyzer", "DataHandler", "ml", "options", "fs"]
