"""
TabularMagic
------------
A package for wide format tabular data.

Provides:
1. Low-code interface for exploratory data analysis and model benchmarking
2. Automatic report generation capabilities


Exposed Classes
---------------
TabularMagic.
    Give a TabularMagic object a DataFrame, get an easy-to-use Python 
    interface for exploring the DataFrame.
DataHandler.
    The data-handling class. 


Submodules
----------
- ml
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



from ._src.tabularmagic import TabularMagic
from ._src.data.datahandler import DataHandler
from . import ml
from . import options



__all__ = ['TabularMagic', 'DataHandler', 'ml', 'options']


