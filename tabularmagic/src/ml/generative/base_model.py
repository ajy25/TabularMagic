import numpy as np



class BaseGenerativeModel():
    """
    BaseGenerativeModel serves as base for generative models. All generative
    models extend BaseGenerativeModel. 

    A generative model learns a probability distribution p(X). All generative 
    models can be fit on a dataset and can be sampled from. 
    """

    def __init__(self):
        pass

    def fit(self, X: np.ndarray):
        """Fits the model.
        
        Parameters
        ----------
        - X : np.ndarray ~ (n_examples, n_features)
        """
        pass

    def sample(self, sample_size: int = None):
        """Samples from the learned distribution.

        Parameters
        ----------
        - sample_size : int. If None, returns a 1d array. 

        Returns
        -------
        - np.ndarray ~ (sample_size, n_features) or (n_features)
        """
        pass


