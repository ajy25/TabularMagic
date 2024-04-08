import numpy as np
from scipy.stats import ecdf


# TODO: Finish this class.


class EmpiricalCDF():
    """Learns a step-wise empirical CDF given 1-dimensional data.
    """

    def __init__(self, data: np.ndarray):
        """
        Parameters
        ----------
        - data : np.ndarray ~ (n,). 
        """
        self._distribution = ecdf(sample=data)
    
    def plot(self):
        pass
